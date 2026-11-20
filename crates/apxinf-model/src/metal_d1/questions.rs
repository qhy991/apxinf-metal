//! D1 question types and request parsing, mirroring the reference's
//! `prompt.py` question classes and `parsing.py`/`as_question` validation.
//!
//! A caller's question is the Decision Index JSON:
//! `{"type": "noul"|"choice"|"score"|"text"|"number"|"date"|"box"|"boxes",
//!   "instructions": str, "criteria"?: map|list, "multiple"?: bool,
//!   "range"?: [low, high], "step"?: x, "unit"?: str, "max"?: n}`.

use apxinf_core::{Error, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(format!("d1 question: {}", message.into()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadKind {
    Text,
    Number,
    Date,
}

impl ReadKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Number => "number",
            Self::Date => "date",
        }
    }

    /// The reference's span cap per read type (`pointer.SPAN_CAPS`).
    pub fn span_cap(self) -> usize {
        match self {
            Self::Text => 512,
            Self::Number | Self::Date => 32,
        }
    }
}

/// A value grid over `[low, high]` read as equally spaced bins (`grid.py`).
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub low: f64,
    pub high: f64,
    pub size: usize,
}

impl Grid {
    pub fn new(low: f64, high: f64, size: usize) -> Result<Self> {
        if size < 2 || !(low < high) {
            return Err(invalid(format!(
                "a grid needs two values or more, low to high: [{low}, {high}] x {size}"
            )));
        }
        Ok(Self { low, high, size })
    }

    pub fn step(&self) -> f64 {
        (self.high - self.low) / (self.size - 1) as f64
    }

    /// The grid values: `low + i * step`, ending exactly on `high`.
    pub fn values(&self) -> Vec<f64> {
        let step = self.step();
        (0..self.size - 1)
            .map(|i| self.low + i as f64 * step)
            .chain([self.high])
            .collect()
    }

    /// Mean, median and central-80% quantiles of a distribution over the bins
    /// (`grid.py Grid.read`); `level` maps a quantile through calibration.
    pub fn read(&self, probabilities: &[f64], level: Option<&dyn Fn(f64) -> f64>) -> GridReadout {
        assert_eq!(probabilities.len(), self.size, "grid readout size");
        let total: f64 = probabilities.iter().sum();
        let values = self.values();
        let mean = values
            .iter()
            .zip(probabilities)
            .map(|(v, p)| v * (p / total))
            .sum::<f64>();
        let quantile = |q: f64, out: &mut Option<f64>| {
            let mut mass = 0.0;
            for (v, p) in values.iter().zip(probabilities) {
                mass += p / total;
                if mass >= q {
                    *out = Some(*v);
                    break;
                }
            }
            // a level near 1 the summed mass falls short of by rounding: the last value
            out.get_or_insert(values[values.len() - 1]);
        };
        let mut p10 = None;
        let mut median = None;
        let mut p90 = None;
        let map = |q: f64| level.map_or(q, |f| f(q));
        quantile(map(0.1), &mut p10);
        quantile(map(0.5), &mut median);
        quantile(map(0.9), &mut p90);
        GridReadout {
            mean,
            p10: p10.unwrap(),
            median: median.unwrap(),
            p90: p90.unwrap(),
        }
    }
}

pub struct GridReadout {
    pub mean: f64,
    pub p10: f64,
    pub median: f64,
    pub p90: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Question {
    /// yes/no: `{"type": "noul", "instructions": ...}` (+ optional
    /// `criteria: {"true": ..., "false": ...}` descriptions).
    Noul {
        instructions: String,
        criteria: Option<(Option<String>, Option<String>)>,
    },
    /// Named options; `multiple` reads one yes/no per option.
    Choice {
        instructions: String,
        criteria: Vec<(String, Option<String>)>,
        multiple: bool,
    },
    /// 2..10 ordered levels, lowest first.
    Score {
        instructions: String,
        levels: Vec<String>,
    },
    /// A value read out of the state (pointer span): text/number/date.
    Read {
        instructions: String,
        kind: ReadKind,
        unit: Option<String>,
    },
    /// A number estimated on a grid over the declared range.
    Estimate {
        instructions: String,
        grid: Grid,
        step: Option<f64>,
        unit: Option<String>,
    },
    /// One object in the single picture.
    Box { instructions: String },
    /// Every instance in reading order; `max` blocks, else 16.
    Boxes { instructions: String, max: Option<usize> },
}

impl Question {
    /// A typed answer (read at its own slots or state positions) — the ones
    /// whose slot probabilities are NOT temperature-calibrated at readout
    /// (`prompt.typed`): reads, estimates, boxes, multiple choices.
    pub fn is_typed(&self) -> bool {
        match self {
            Self::Read { .. } | Self::Estimate { .. } | Self::Box { .. } | Self::Boxes { .. } => true,
            Self::Choice { multiple, .. } => *multiple,
            Self::Noul { .. } | Self::Score { .. } => false,
        }
    }

    /// The `boxes` block count (`box.INSTANCES` when no `max`).
    pub fn blocks(&self) -> usize {
        match self {
            Self::Boxes { max, .. } => max.unwrap_or(16),
            _ => 1,
        }
    }
}

fn instructions_of(q: &serde_json::Value) -> Result<String> {
    match q.get("instructions") {
        Some(serde_json::Value::String(text)) if !text.trim().is_empty() => Ok(text.clone()),
        Some(_) => Err(invalid(
            "\"instructions\" is the question as text (spoken instructions are unsupported)",
        )),
        None => Err(invalid("\"instructions\" is required")),
    }
}

/// Parse one question dict with the reference's validation (`check_question`).
pub fn from_json(q: &serde_json::Value) -> Result<Question> {
    let obj = q
        .as_object()
        .ok_or_else(|| invalid("a question is a JSON object"))?;
    for key in obj.keys() {
        if !matches!(
            key.as_str(),
            "type" | "instructions" | "criteria" | "multiple" | "range" | "step" | "unit" | "max"
        ) {
            return Err(invalid(format!("unknown field `{key}`")));
        }
    }
    let kind = obj
        .get("type")
        .and_then(|k| k.as_str())
        .ok_or_else(|| invalid("\"type\" is required"))?;
    let instructions = instructions_of(q)?;
    let multiple = obj
        .get("multiple")
        .and_then(|m| m.as_bool())
        .unwrap_or(false);
    if multiple && kind != "choice" {
        return Err(invalid("\"multiple\" belongs to a choice"));
    }
    if obj.contains_key("step") && !obj.contains_key("range") {
        return Err(invalid("\"step\" needs a \"range\""));
    }
    let unit = obj
        .get("unit")
        .and_then(|u| u.as_str())
        .map(str::to_string);
    match kind {
        "noul" => {
            let criteria = match obj.get("criteria") {
                None => None,
                Some(serde_json::Value::Object(map)) => {
                    let get = |k: &str| map.get(k).and_then(|v| v.as_str()).map(str::to_string);
                    Some((get("true"), get("false")))
                }
                Some(_) => {
                    return Err(invalid(
                        "a noul's \"criteria\", when given, is {\"true\": ..., \"false\": ...}",
                    ))
                }
            };
            Ok(Question::Noul {
                instructions,
                criteria,
            })
        }
        "choice" => {
            let map = match obj.get("criteria") {
                Some(serde_json::Value::Object(map)) if !map.is_empty() => map,
                _ => {
                    return Err(invalid(
                        "a choice's \"criteria\" names the options: {\"a\": \"desc\", ...}",
                    ))
                }
            };
            let mut criteria = Vec::with_capacity(map.len());
            for (name, desc) in map {
                let desc = match desc {
                    serde_json::Value::Null => None,
                    serde_json::Value::String(text) => Some(text.clone()),
                    _ => return Err(invalid("a choice's descriptions are strings or null")),
                };
                criteria.push((name.clone(), desc));
            }
            Ok(Question::Choice {
                instructions,
                criteria,
                multiple,
            })
        }
        "score" => {
            let list = match obj.get("criteria") {
                Some(serde_json::Value::Array(list)) => list,
                _ => return Err(invalid("a score's \"criteria\" lists 2 to 10 levels")),
            };
            if list.len() < 2 || list.len() > 10 {
                return Err(invalid("a score's \"criteria\" lists 2 to 10 levels"));
            }
            let mut levels = Vec::with_capacity(list.len());
            for level in list {
                levels.push(
                    level
                        .as_str()
                        .ok_or_else(|| invalid("a score's levels are strings"))?
                        .to_string(),
                );
            }
            Ok(Question::Score {
                instructions,
                levels,
            })
        }
        "text" | "date" => Ok(Question::Read {
            instructions,
            kind: if kind == "text" {
                ReadKind::Text
            } else {
                ReadKind::Date
            },
            unit,
        }),
        "number" => {
            if let Some(bounds) = obj.get("range") {
                let bounds = bounds
                    .as_array()
                    .filter(|b| b.len() == 2)
                    .ok_or_else(|| invalid("an estimate's \"range\" is [low, high]"))?;
                let low = bounds[0]
                    .as_f64()
                    .ok_or_else(|| invalid("an estimate's range bounds are numbers"))?;
                let high = bounds[1]
                    .as_f64()
                    .ok_or_else(|| invalid("an estimate's range bounds are numbers"))?;
                let step = match obj.get("step") {
                    None => None,
                    Some(s) => Some(
                        s.as_f64()
                            .ok_or_else(|| invalid("an estimate's step is a number"))?,
                    ),
                };
                let step_divides = |s: f64| {
                    let r = (high - low) / s;
                    s > 0.0 && (r - r.round()).abs() <= 1e-9 * r.abs().max(1e-12)
                };
                if !low.is_finite()
                    || !high.is_finite()
                    || !low.lt(&high)
                    || step.is_some_and(|s| !step_divides(s))
                {
                    return Err(invalid(
                        "an estimate's range runs low to high and its step, above 0, divides it",
                    ));
                }
                // `Estimate.grid`: 128 values, or `step`-many when fewer.
                let size = match step {
                    None => 128,
                    Some(s) => (((high - low) / s).round() as usize + 1).min(128),
                };
                Ok(Question::Estimate {
                    instructions,
                    grid: Grid::new(low, high, size)?,
                    step,
                    unit,
                })
            } else {
                Ok(Question::Read {
                    instructions,
                    kind: ReadKind::Number,
                    unit,
                })
            }
        }
        "box" => Ok(Question::Box { instructions }),
        "boxes" => {
            let max = match obj.get("max") {
                None => None,
                Some(m) => {
                    let max = m
                        .as_u64()
                        .ok_or_else(|| invalid("\"max\" is a whole number"))?;
                    if !(1..=64).contains(&max) {
                        return Err(invalid("\"max\" is a whole number from 1 to 64"));
                    }
                    Some(max as usize)
                }
            };
            Ok(Question::Boxes { instructions, max })
        }
        other => Err(invalid(format!(
            "\"type\" is one of noul, choice, score, text, number, date, box, boxes, not {other:?}"
        ))),
    }
}
