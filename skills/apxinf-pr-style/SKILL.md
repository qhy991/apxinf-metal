---
name: apxinf-pr-style
description: >-
  Prepare a change for review in apxinf (or a fork) so the diff reads as engine
  code rather than experiment history. Use before opening or updating a pull
  request: auditing comments for narration, naming, commit subjects, PR
  description structure, test and provenance requirements, and which parts of a
  working branch must not be submitted at all — including Chinese asks like
  "这个 PR 干净吗", "注释里有没有实验痕迹", "该怎么写 PR 描述".
---

# apxinf-pr-style

**Goal:** the reviewer sees a contract and its evidence, never the path that got
there. A working tree accumulates experiment notes, machine-specific numbers and
half-finished switches; a PR is the smallest change that stands on its own.

Run [`check-pr.sh`](check-pr.sh) on the branch before every push. It only catches
text patterns; the judgment below is still yours.

## What must not appear in a diff

| Never | Why | Instead |
| --- | --- | --- |
| Experiment narrative ("first measured at 1.107x", "we tried X") | Reads as a lab notebook; rots immediately | Put the number in the PR description, with machine and command |
| References to a demo, game, campaign or private branch by name | Couples engine code to one consumer | Describe the capability: "media-conditioned decisions", "tiled ANE handoff" |
| Machine names, hostnames, user paths (`/Users/...`) | Not reproducible for a reviewer; leaks local setup | Environment variable, or nothing |
| Dates and "as of <commit>" markers in comments | Dated claims age badly | Keep provenance in `NOTICE` / commit message |
| `TODO`, `FIXME`, `XXX`, `TEMP`, `hack`, disabled code | Unfinished surface in a review | Remove it; open an issue instead |
| Commented-out code | Reviewers cannot tell intent from residue | Delete it, Git remembers |
| Chinese (or mixed-language) comments in source | The upstream tree is English-only | English |
| Per-op wrapper functions added "so the demo can call it" | AGENTS.md: direct calls to selected kernels are valid | Call the kernel directly |

## What a comment is for

1. **The contract** a caller cannot infer: shapes, dtypes, what is rejected and
   why, rounding points, ownership of buffers.
2. **The unsafe justification** — `// SAFETY:` immediately above the block, naming
   the bounds and the type invariants it relies on.
3. **Provenance** when the code is adapted: one line naming upstream, version and
   licence, ending in `See NOTICE`. The kernel header is the right place.

A comment that explains *why the old code was slow*, or *what this replaces*, is
history. Keep the invariant, drop the comparison.

```rust
// Good: the caller cannot infer this.
/// Rejects any palette other than 4 or 6 bits, and any index outside the palette.

// Bad: narrates the alternative implementation.
// One gather kernel replaces the three strided gathers the consumers would
// otherwise each do (RMSNorm set_output, and the recurrence's input copy).
```

## Commit messages

`type(scope): imperative subject` — `feat`, `fix`, `perf`, `refactor`, `docs`,
`test`, `build`, `ci`, `chore`. Subject ≤ 72 characters, no trailing period.
Scope is the crate or the subsystem, not the consumer (`mlx`, `coreml`,
`clef`, `serving`).

Body: what changed and why it is safe, in that order. **No measurement claims** —
"verified on <machine>, 17 passed" belongs in the PR description, where it can be
re-run and re-read. No `Co-Authored-By` unless the work was genuinely joint.

One commit per reviewable idea. A test added for a change lands in the same PR;
it does not need its own commit unless it stands alone.

## PR description

Keep this order — reviewers read the first paragraph and skim the rest:

1. **What** the change is, one paragraph, no history.
2. **Why it is correct / safe** — invariants, rounding, fallback paths.
3. **Scope** — what is deliberately *not* included, and what it depends on.
4. **Testing** — commands, expected output, hardware used.
5. **Relationship to open PRs** — shared files, merge order, rebase expectation.

Then the amount of code the reviewer must read. If a section is empty, say so
explicitly ("no behaviour change; nothing calls this yet") rather than omitting it.

## What stays out of the PR

- Demo glue, harness scripts, recording or evaluation tooling, prompt text.
- Experiment switches that were only used to produce a number (a named environment
  variable that selects a *behaviour* may stay, if the PR description explains it).
- Weights, caches, traces, generated packages.
- Anything the PR does not test or cannot be re-derived by the reviewer.

If a capability is only reachable through one private consumer, that is a signal
the boundary is wrong: submit the engine capability, keep the consumer outside.

## Pre-submit checklist

```sh
./check-pr.sh                 # text audit of the diff and commit subjects
```

- [ ] `cargo check` / `cargo test` for every crate the diff touches
- [ ] At least one test exercises the new path, or the description says why none can
- [ ] Comments state contracts, not history; no machine names, dates or TODOs
- [ ] Commit subjects match `type(scope): …` and carry no measurement claims
- [ ] PR description has What / Why-safe / Scope / Testing / Relationship
- [ ] Adapted code carries a provenance line and the licence is in `NOTICE`
- [ ] The diff contains no demo, harness or recorded artifact
