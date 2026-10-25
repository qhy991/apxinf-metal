use crate::llm_trait::{GeneratedToken, TokenSequence};
use apxinf_core::{Error, Result, TokenPenalties, TokenSamplingParams, TokenSelection};
use apxinf_mlx::{greedy_token_arrays, Array, Compiled, MlxDType, Stream};

pub(crate) struct CheckedGreedy {
    state: GreedyState,
}

enum GreedyState {
    Eager,
    Requested(GreedyProfile),
    Prepared {
        profile: GreedyProfile,
        callable: Compiled,
    },
}

#[derive(Clone, Copy)]
struct GreedyProfile {
    vocab: usize,
    dtype: MlxDType,
}

impl CheckedGreedy {
    pub(crate) fn for_model(vocab: usize, dtype: MlxDType, eligible: bool) -> Result<Self> {
        if !eligible {
            return Ok(Self {
                state: GreedyState::Eager,
            });
        }
        Self::from_override(
            vocab,
            dtype,
            std::env::var_os("APXINF_MLX_COMPILED_GREEDY").as_deref(),
        )
    }

    fn from_override(
        vocab: usize,
        dtype: MlxDType,
        value: Option<&std::ffi::OsStr>,
    ) -> Result<Self> {
        let state = if value.is_some_and(|value| value == "1") {
            if vocab == 0
                || vocab > i32::MAX as usize
                || !matches!(dtype, MlxDType::F32 | MlxDType::BF16)
            {
                return Err(Error::Contract("invalid compiled greedy profile"));
            }
            GreedyState::Requested(GreedyProfile { vocab, dtype })
        } else {
            GreedyState::Eager
        };
        Ok(Self { state })
    }

    pub(crate) fn prepare(&mut self, stream: &Stream) -> Result<()> {
        if !matches!(self.state, GreedyState::Requested(_)) {
            return Ok(());
        }
        let callable = Compiled::new(stream, 2, |inputs| {
            let [row] = inputs else {
                return Err(Error::Contract("compiled greedy requires one logits row"));
            };
            let (feed, checked) = greedy_token_arrays(row)?;
            Ok(vec![feed, checked])
        })?;
        self.prepare_callable(stream, callable)
    }

    fn prepare_callable(&mut self, stream: &Stream, callable: Compiled) -> Result<()> {
        let profile = match &self.state {
            GreedyState::Requested(profile) => *profile,
            _ => {
                return Err(Error::Contract(
                    "compiled greedy preparation was not requested",
                ))
            }
        };
        let probe = Array::zeros(stream, &[1, profile.vocab], profile.dtype)?;
        let result = callable
            .prepare(std::slice::from_ref(&probe))
            .and_then(Self::outputs)
            .and_then(|(_, checked)| checked.to_bytes_ready().map(|_| ()));
        if let Err(error) = result {
            return match stream.synchronize() {
                Ok(()) => Err(error),
                Err(drain) => Err(Error::Other(format!(
                    "compiled greedy preparation failed: {error}; stream cleanup failed: {drain}"
                ))),
            };
        }
        self.state = GreedyState::Prepared { profile, callable };
        Ok(())
    }

    pub(crate) fn is_ready(&self) -> bool {
        !matches!(self.state, GreedyState::Requested(_))
    }

    pub(crate) fn arrays(&self, logits: &Array) -> Result<(Array, Array)> {
        let (profile, callable) = match &self.state {
            GreedyState::Eager => return greedy_token_arrays(logits),
            GreedyState::Requested(_) => {
                return Err(Error::Contract("prepare compiled greedy before inference"));
            }
            GreedyState::Prepared { profile, callable } => (profile, callable),
        };
        if logits.shape().last() != Some(&profile.vocab)
            || logits.numel() == 0
            || logits.dtype() != profile.dtype
        {
            return Err(Error::Contract(
                "logits differ from the prepared greedy profile",
            ));
        }
        let rows = logits.numel() / profile.vocab;
        let row = logits
            .reshape(&[rows, profile.vocab])?
            .slice_axis(0, rows - 1, rows)?;
        let traces = logits.stream().counters().trace_callbacks;
        let outputs = callable.call(&[row])?;
        if logits.stream().counters().trace_callbacks != traces {
            return Err(Error::Contract("compiled greedy retraced during inference"));
        }
        Self::outputs(outputs)
    }

    fn outputs(outputs: Vec<Array>) -> Result<(Array, Array)> {
        let [feed, checked]: [Array; 2] = outputs
            .try_into()
            .map_err(|_| Error::Contract("compiled greedy requires two outputs"))?;
        if [&feed, &checked]
            .iter()
            .any(|array| array.shape() != [1] || array.dtype() != MlxDType::U32)
        {
            return Err(Error::Contract(
                "compiled greedy outputs must each be one u32",
            ));
        }
        Ok((feed, checked))
    }
}

pub(crate) fn enabled(params: &TokenSamplingParams) -> bool {
    params.selection == TokenSelection::Greedy
        && params.penalties == TokenPenalties::default()
        && !params.return_logprob
        && std::env::var_os("APXINF_MLX_SYNC_DECODE").is_none_or(|value| value != "1")
}

pub(crate) enum DecodeFinish {
    PublishedPosition(usize),
    Invalidated,
}

pub(crate) trait DeviceDecode {
    fn stream(&self) -> &Stream;
    fn greedy_arrays(&self, logits: &Array) -> Result<(Array, Array)> {
        greedy_token_arrays(logits)
    }
    /// Evaluating the returned logits must evaluate every state array updated by this step.
    fn decode(&mut self, token: &Array) -> Result<Array>;
    fn finish_decode(&mut self, outcome: DecodeFinish) -> Result<()>;
}

struct PendingToken {
    feed: Array,
    checked: Array,
}

impl PendingToken {
    fn new((feed, checked): (Array, Array)) -> Self {
        Self { feed, checked }
    }

    fn read(&self) -> Result<GeneratedToken> {
        let bytes = self.checked.to_bytes_ready()?;
        let token_id = u32::from_ne_bytes(
            bytes
                .try_into()
                .map_err(|_| Error::Contract("greedy token must contain one u32"))?,
        );
        if token_id == u32::MAX {
            return Err(Error::Other("all token logits are invalid".into()));
        }
        Ok(GeneratedToken {
            token_id,
            logprob: None,
        })
    }
}

pub(crate) struct GreedySequence<'a, M: DeviceDecode> {
    model: &'a mut M,
    current: Option<PendingToken>,
    successor: Option<PendingToken>,
    prompt_end: usize,
    failed: bool,
    finished: bool,
}

impl<'a, M: DeviceDecode> GreedySequence<'a, M> {
    pub(crate) fn new(model: &'a mut M, first: Array, prompt_end: u32) -> Result<Self> {
        let mut sequence = Self {
            model,
            current: None,
            successor: None,
            prompt_end: prompt_end as usize,
            failed: false,
            finished: false,
        };
        sequence.current = Some(PendingToken::new(sequence.model.greedy_arrays(&first)?));
        sequence.model.stream().async_eval(&[sequence
            .current
            .as_ref()
            .unwrap()
            .checked
            .clone()])?;
        Ok(sequence)
    }

    fn advance(&mut self, lookahead: bool) -> Result<GeneratedToken> {
        let current = self
            .current
            .as_ref()
            .ok_or(Error::Contract("token sequence exhausted"))?;
        if lookahead {
            let logits = self.model.decode(&current.feed)?;
            self.successor = Some(PendingToken::new(self.model.greedy_arrays(&logits)?));
            let outputs = [self.successor.as_ref().unwrap().checked.clone()];
            self.model.stream().async_eval(&outputs)?;
        }
        let token = current.read()?;
        self.current = self.successor.take();
        Ok(token)
    }
}

impl<M: DeviceDecode> TokenSequence for GreedySequence<'_, M> {
    fn next(&mut self, lookahead: bool) -> Result<GeneratedToken> {
        let result = self.advance(lookahead);
        self.failed |= result.is_err();
        result
    }

    fn finish(mut self: Box<Self>, published: usize) -> Result<()> {
        let outcome = if self.failed {
            DecodeFinish::Invalidated
        } else {
            DecodeFinish::PublishedPosition(self.prompt_end + published.saturating_sub(1))
        };
        let result = self.model.finish_decode(outcome);
        self.finished = true;
        result
    }

    fn overlaps_callbacks(&self) -> bool {
        true
    }
}

impl<M: DeviceDecode> Drop for GreedySequence<'_, M> {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.model.finish_decode(DecodeFinish::Invalidated);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GenerationCancellation, GenerationOptions, LlmInput, LlmTrait, LoadedModel};
    use apxinf_core::{Backend, CpuBackend, Device, Tensor, TokenSampler};
    use apxinf_loader::ModelConfig;
    use std::{cell::RefCell, collections::HashMap, rc::Rc};

    fn requested(vocab: usize, dtype: MlxDType) -> CheckedGreedy {
        CheckedGreedy::from_override(vocab, dtype, Some(std::ffi::OsStr::new("1"))).unwrap()
    }

    fn read_pair(stream: &Stream, pair: &(Array, Array)) -> (u32, u32) {
        stream
            .async_eval(&[pair.0.clone(), pair.1.clone()])
            .unwrap();
        let checked = u32::from_ne_bytes(pair.1.to_bytes_ready().unwrap().try_into().unwrap());
        (pair.0.to_u32_scalar().unwrap(), checked)
    }

    fn semantic_cases(stream: &Stream) {
        for dtype in [MlxDType::F32, MlxDType::BF16] {
            let mut sampler = requested(4, dtype);
            assert!(!sampler.is_ready());
            sampler.prepare(stream).unwrap();
            assert!(sampler.is_ready());
            let prepared = stream.counters();
            sampler.prepare(stream).unwrap();
            assert_eq!(stream.counters(), prepared);
            for (values, expected) in [
                ([f32::NAN, 4., 4., f32::NEG_INFINITY], (1, 1)),
                ([-0., 0., -1., -2.], (0, 0)),
                ([1., f32::INFINITY, 2., f32::INFINITY], (1, 1)),
                ([f32::MAX, f32::INFINITY, 0., -1.], (0, 0)),
                ([-5., -2., -2., -3.], (1, 1)),
                (
                    [f32::NAN, f32::NEG_INFINITY, f32::NAN, f32::NEG_INFINITY],
                    (0, u32::MAX),
                ),
            ] {
                let logits = Array::from_f32(stream, &[1, 4], &values)
                    .unwrap()
                    .cast(dtype)
                    .unwrap();
                let eager = greedy_token_arrays(&logits).unwrap();
                assert_eq!(read_pair(stream, &eager), expected);
                let before = stream.counters();
                let compiled = sampler.arrays(&logits).unwrap();
                assert_eq!(stream.counters(), before);
                assert_eq!(read_pair(stream, &compiled), expected);
                assert_eq!(stream.counters().trace_callbacks, prepared.trace_callbacks);
            }
            let logits = Array::from_f32(stream, &[2, 4], &[9., 0., 0., 0., 0., 1., 8., 0.])
                .unwrap()
                .cast(dtype)
                .unwrap();
            for shape in [&[4][..], &[1, 4], &[1, 1, 4], &[1, 1, 1, 4]] {
                let last = logits.slice_axis(0, 1, 2).unwrap().reshape(shape).unwrap();
                assert_eq!(read_pair(stream, &sampler.arrays(&last).unwrap()), (2, 2));
            }
            for shape in [&[2, 4][..], &[1, 2, 4], &[1, 1, 2, 4]] {
                let rows = logits.reshape(shape).unwrap();
                assert_eq!(read_pair(stream, &sampler.arrays(&rows).unwrap()), (2, 2));
            }
            let storage = Array::from_f32(
                stream,
                &[12],
                &[0., 9., 0., 0., 0., 0., 0., 0., 1., 8., 0., 0.],
            )
            .unwrap()
            .cast(dtype)
            .unwrap();
            let offset = storage.as_strided(&[2, 4], &[6, 1], 1).unwrap();
            let transposed = Array::from_f32(stream, &[4, 2], &[9., 0., 0., 1., 0., 8., 0., 0.])
                .unwrap()
                .cast(dtype)
                .unwrap()
                .transpose(&[1, 0])
                .unwrap();
            for view in [offset, transposed] {
                assert_eq!(read_pair(stream, &sampler.arrays(&view).unwrap()), (2, 2));
            }
            let retained = sampler.arrays(&logits).unwrap();
            let changed = Array::from_f32(stream, &[4], &[0., 9., 0., 0.])
                .unwrap()
                .cast(dtype)
                .unwrap();
            assert_eq!(
                read_pair(stream, &sampler.arrays(&changed).unwrap()),
                (1, 1)
            );
            assert_eq!(read_pair(stream, &retained), (2, 2));
            assert_eq!(stream.counters().trace_callbacks, prepared.trace_callbacks);
            drop(sampler);
            assert_eq!(read_pair(stream, &retained), (2, 2));
        }
        let mut singleton = requested(1, MlxDType::BF16);
        singleton.prepare(stream).unwrap();
        let one = Array::zeros(stream, &[1], MlxDType::BF16).unwrap();
        assert_eq!(read_pair(stream, &singleton.arrays(&one).unwrap()), (0, 0));
    }

    fn profile_cases(stream: &Stream, foreign: &Stream) {
        let mut first = requested(4, MlxDType::F32);
        let input = Array::from_f32(stream, &[4], &[0., 2., 0., 0.]).unwrap();
        assert!(first.arrays(&input).is_err());
        first.prepare(stream).unwrap();
        let mut second = requested(4, MlxDType::F32);
        second.prepare(stream).unwrap();
        let traces = stream.counters().trace_callbacks;
        for wrong in [
            Array::zeros(stream, &[0, 4], MlxDType::F32).unwrap(),
            Array::zeros(stream, &[1, 3], MlxDType::F32).unwrap(),
            Array::zeros(stream, &[], MlxDType::F32).unwrap(),
            input.cast(MlxDType::BF16).unwrap(),
            Array::zeros(foreign, &[1, 4], MlxDType::F32).unwrap(),
        ] {
            assert!(first.arrays(&wrong).is_err());
            assert_eq!(stream.counters().trace_callbacks, traces);
        }
        assert_eq!(read_pair(stream, &second.arrays(&input).unwrap()), (1, 1));
        drop(first);
        assert_eq!(read_pair(stream, &second.arrays(&input).unwrap()), (1, 1));
        assert_eq!(stream.counters().trace_callbacks, traces);

        let callable = Compiled::new(stream, 2, |a| {
            let (feed, checked) = greedy_token_arrays(&a[0])?;
            Ok(vec![feed, checked])
        })
        .unwrap();
        let untraced = CheckedGreedy {
            state: GreedyState::Prepared {
                profile: GreedyProfile {
                    vocab: 4,
                    dtype: MlxDType::F32,
                },
                callable,
            },
        };
        assert!(untraced
            .arrays(&input)
            .unwrap_err()
            .to_string()
            .contains("retraced"));
        assert!(stream.counters().trace_callbacks > traces);
        stream.synchronize().unwrap();
    }

    fn failed_preparation_cases(stream: &Stream) {
        let mut sampler = requested(4, MlxDType::F32);
        let marker = Rc::new(());
        let weak = Rc::downgrade(&marker);
        let bad = Compiled::new(stream, 2, move |a| {
            let _keep = &marker;
            let (feed, checked) = greedy_token_arrays(&a[0])?;
            Ok(vec![feed, checked.cast(MlxDType::F32)?])
        })
        .unwrap();
        let before = stream.counters().eval_calls;
        assert!(sampler.prepare_callable(stream, bad).is_err());
        assert!(stream.counters().eval_calls >= before + 2);
        assert!(!sampler.is_ready());
        assert!(weak.upgrade().is_none());
        let input = Array::from_f32(stream, &[4], &[0., 0., 7., 0.]).unwrap();
        assert!(sampler.arrays(&input).is_err());
        let bad = Compiled::new(stream, 2, |_| {
            Err(Error::Contract("injected preparation failure"))
        })
        .unwrap();
        assert!(sampler.prepare_callable(stream, bad).is_err());
        assert!(!sampler.is_ready());
        sampler.prepare(stream).unwrap();
        assert!(sampler.is_ready());
        assert_eq!(read_pair(stream, &sampler.arrays(&input).unwrap()), (2, 2));
    }

    #[test]
    fn compiled_greedy_selection_is_exact_and_default_off() {
        for value in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some("01"),
            Some(" 1"),
        ] {
            let sampler =
                CheckedGreedy::from_override(4, MlxDType::F32, value.map(std::ffi::OsStr::new))
                    .unwrap();
            assert!(matches!(sampler.state, GreedyState::Eager));
        }
        assert!(!requested(4, MlxDType::F32).is_ready());
        for (vocab, dtype) in [
            (0, MlxDType::F32),
            (i32::MAX as usize + 1, MlxDType::BF16),
            (4, MlxDType::U32),
        ] {
            assert!(
                CheckedGreedy::from_override(vocab, dtype, Some(std::ffi::OsStr::new("1")))
                    .is_err()
            );
            assert!(CheckedGreedy::for_model(vocab, dtype, false)
                .unwrap()
                .is_ready());
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(CheckedGreedy::from_override(
                4,
                MlxDType::F32,
                Some(std::ffi::OsStr::from_bytes(&[0xff]))
            )
            .unwrap()
            .is_ready());
        }
    }

    #[test]
    fn compiled_greedy_cpu_semantics_and_retained_outputs() {
        semantic_cases(&Stream::cpu().unwrap());
    }

    #[test]
    fn compiled_greedy_cpu_profiles_streams_and_retrace_rejection() {
        profile_cases(&Stream::cpu().unwrap(), &Stream::cpu().unwrap());
    }

    #[test]
    fn compiled_greedy_cpu_failed_preparation_releases_and_retries() {
        failed_preparation_cases(&Stream::cpu().unwrap());
    }

    #[test]
    #[ignore = "requires exclusive shared Metal locks and pinned MLX runtime"]
    fn compiled_greedy_metal_semantics_profiles_and_recovery() {
        let stream = Stream::metal(0).unwrap();
        semantic_cases(&stream);
        profile_cases(&stream, &Stream::metal(0).unwrap());
        failed_preparation_cases(&stream);
    }

    #[test]
    fn cpu_pipeline_sampling_failures_and_abandonment_invalidate_once() {
        for compiled in [false, true] {
            for position in [2, 3] {
                let mut model = test_model(vec![Some(2), Some(3)], false, compiled);
                model.fail_sample_at = Some(position);
                let observed = model.observed.clone();
                let mut loaded = LoadedModel::text(Box::new(model));
                let error = loaded
                    .generate_streaming_with_options(
                        LlmInput::text(&[0, 1]),
                        &GenerationOptions::greedy(8, Some(2)),
                        |_| panic!("token published after sampling failure"),
                    )
                    .err()
                    .unwrap();
                assert!(error.to_string().contains("injected sampling failure"));
                let state = observed.borrow();
                assert_eq!(state.finishes, vec![None]);
                assert_eq!(state.enqueues.len(), position - 2);
            }
            let mut model = test_model(vec![Some(2), Some(3)], false, compiled);
            model.prefill(LlmInput::text(&[0, 1])).unwrap();
            let first = model.logits(0).unwrap();
            let observed = model.observed.clone();
            let mut sequence = GreedySequence::new(&mut model, first, 2).unwrap();
            assert_eq!(sequence.next(true).unwrap().token_id, 2);
            drop(sequence);
            assert_eq!(observed.borrow().finishes, vec![None]);
        }
    }

    #[derive(Default)]
    struct Observed {
        position: usize,
        enqueues: Vec<(usize, usize)>,
        finishes: Vec<Option<usize>>,
        resets: usize,
    }

    struct Model {
        stream: Stream,
        observed: Rc<RefCell<Observed>>,
        prompt: usize,
        tokens: Vec<Option<u32>>,
        fail_decode: bool,
        greedy: CheckedGreedy,
        download_baseline: usize,
        fail_sample_at: Option<usize>,
    }

    impl Model {
        fn logits(&self, index: usize) -> Result<Array> {
            let mut values = [f32::NEG_INFINITY; 4];
            if let Some(token) = self.tokens[index.min(self.tokens.len() - 1)] {
                values[token as usize] = 1.;
            }
            Array::from_f32(&self.stream, &[1, 4], &values)
        }
    }

    impl DeviceDecode for Model {
        fn stream(&self) -> &Stream {
            &self.stream
        }
        fn greedy_arrays(&self, logits: &Array) -> Result<(Array, Array)> {
            if self.fail_sample_at == Some(self.observed.borrow().position) {
                return Err(Error::Other("injected sampling failure".into()));
            }
            self.greedy.arrays(logits)
        }
        fn decode(&mut self, _token: &Array) -> Result<Array> {
            let position = self.observed.borrow().position;
            self.observed.borrow_mut().enqueues.push((
                position,
                self.stream.counters().downloads_bytes - self.download_baseline,
            ));
            if self.fail_decode {
                return Err(Error::Other("injected decode failure".into()));
            }
            self.observed.borrow_mut().position += 1;
            self.logits(position - self.prompt + 1)
        }
        fn finish_decode(&mut self, outcome: DecodeFinish) -> Result<()> {
            self.stream.synchronize()?;
            let position = match outcome {
                DecodeFinish::PublishedPosition(position) => Some(position),
                DecodeFinish::Invalidated => None,
            };
            let mut observed = self.observed.borrow_mut();
            observed.finishes.push(position);
            if let Some(position) = position {
                assert!(position <= observed.position && observed.position - position <= 1);
                observed.position = position;
            }
            Ok(())
        }
    }

    impl LlmTrait for Model {
        fn load(_: ModelConfig, _: HashMap<String, Tensor>, _: Device) -> Result<Self> {
            unreachable!()
        }
        fn forward(&mut self, _: &[u32], _: u32) -> Result<Tensor> {
            unreachable!()
        }
        fn backend(&self) -> &dyn Backend {
            &CpuBackend
        }
        fn reset(&mut self) {
            self.observed.borrow_mut().resets += 1;
        }
        fn vocab_size(&self) -> usize {
            4
        }
        fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
            self.prompt = input.token_ids.len();
            self.observed.borrow_mut().position = self.prompt;
            Tensor::from_f32(vec![1, 4], &[0.; 4])
        }
        fn token_sequence<'a>(
            &'a mut self,
            _: Tensor,
            _: Box<dyn TokenSampler>,
            _: &TokenSamplingParams,
            prompt_end: u32,
        ) -> Result<Box<dyn TokenSequence + 'a>> {
            let first = self.logits(0)?;
            Ok(Box::new(GreedySequence::new(self, first, prompt_end)?))
        }
    }

    fn test_model(tokens: Vec<Option<u32>>, fail_decode: bool, compiled: bool) -> Model {
        let stream = Stream::cpu().unwrap();
        let mut greedy = CheckedGreedy::from_override(
            4,
            MlxDType::F32,
            compiled.then_some(std::ffi::OsStr::new("1")),
        )
        .unwrap();
        greedy.prepare(&stream).unwrap();
        let download_baseline = stream.counters().downloads_bytes;
        Model {
            stream,
            observed: Rc::new(RefCell::new(Observed::default())),
            prompt: 0,
            tokens,
            fail_decode,
            greedy,
            download_baseline,
            fail_sample_at: None,
        }
    }

    fn loaded(
        tokens: Vec<Option<u32>>,
        fail_decode: bool,
        compiled: bool,
    ) -> (LoadedModel, Rc<RefCell<Observed>>) {
        let model = test_model(tokens, fail_decode, compiled);
        let observed = model.observed.clone();
        (LoadedModel::text(Box::new(model)), observed)
    }

    #[test]
    fn cpu_pipeline_submits_before_read_and_stops_at_budget() {
        for compiled in [false, true] {
            let (mut model, observed) = loaded(vec![Some(2), Some(3), Some(1)], false, compiled);
            let mut callbacks = vec![];
            let output = model
                .generate_streaming_with_options(
                    LlmInput::text(&[0, 1]),
                    &GenerationOptions::greedy(3, None),
                    |token| callbacks.push(token.token_id),
                )
                .unwrap();
            assert_eq!(output.token_ids(), vec![2, 3, 1]);
            assert_eq!(callbacks, output.token_ids());
            let state = observed.borrow();
            assert_eq!(state.enqueues, vec![(2, 0), (3, 4)]);
            assert_eq!(state.finishes, vec![Some(4)]);
            assert_eq!(state.position, 4);
        }
    }

    #[test]
    fn cpu_pipeline_eos_discards_one_step_and_ignores_unused_invalid_logits() {
        for compiled in [false, true] {
            for eos in [2, 3] {
                let tokens = if eos == 2 {
                    vec![Some(2), None]
                } else {
                    vec![Some(2), Some(3), None]
                };
                let (mut model, observed) = loaded(tokens, false, compiled);
                let output = model
                    .generate_streaming_with_options(
                        LlmInput::text(&[0, 1]),
                        &GenerationOptions::greedy(8, Some(eos)),
                        |_| {},
                    )
                    .unwrap();
                let expected = if eos == 2 { vec![2] } else { vec![2, 3] };
                assert_eq!(output.token_ids(), expected);
                let state = observed.borrow();
                assert_eq!(state.enqueues.len(), expected.len());
                assert_eq!(state.finishes, vec![Some(2 + expected.len() - 1)]);
                assert_eq!(state.position, 2 + expected.len() - 1);
            }
        }
    }

    #[test]
    fn cpu_pipeline_cancel_returns_only_published_prefix() {
        for compiled in [false, true] {
            let (mut model, observed) = loaded(vec![Some(2), Some(3), Some(1)], false, compiled);
            let cancel = GenerationCancellation::default();
            let mut callbacks = vec![];
            let output = model
                .generate_streaming_cancellable(
                    LlmInput::text(&[0, 1]),
                    &GenerationOptions::greedy(8, None),
                    &cancel,
                    |token| {
                        callbacks.push(token.token_id);
                        let external = cancel.clone();
                        std::thread::spawn(move || external.cancel())
                            .join()
                            .unwrap();
                    },
                )
                .unwrap();
            assert_eq!(callbacks, vec![2]);
            assert_eq!(output.token_ids(), callbacks);
            assert_eq!(observed.borrow().finishes, vec![Some(2)]);
            let output = model
                .generate_streaming_cancellable(
                    LlmInput::text(&[0, 1]),
                    &GenerationOptions::greedy(8, None),
                    &cancel,
                    |_| panic!("cancelled callback"),
                )
                .unwrap();
            assert!(output.tokens.is_empty());
            assert_eq!(observed.borrow().resets, 1);
        }
    }

    #[test]
    fn cpu_pipeline_errors_drain_and_invalidate_once() {
        for compiled in [false, true] {
            for fail_decode in [false, true] {
                let (mut model, observed) = loaded(vec![None, Some(2)], fail_decode, compiled);
                let error = model
                    .generate_streaming_with_options(
                        LlmInput::text(&[0, 1]),
                        &GenerationOptions::greedy(8, None),
                        |_| panic!("invalid token published"),
                    )
                    .err()
                    .unwrap();
                assert!(error.to_string().contains(if fail_decode {
                    "injected decode failure"
                } else {
                    "all token logits are invalid"
                }));
                assert_eq!(observed.borrow().finishes, vec![None]);
            }
        }
    }
}
