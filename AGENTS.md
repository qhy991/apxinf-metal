# ApxInf project rules

## Scope

These rules apply to new first-party work in this repository.
Keep existing dependency and vendor records intact.

## Independent implementation

- Write all new implementation code by hand.
- Write all new tests and fixtures by hand.
- Use external repositories to study behavior, interfaces, and design choices.
- Do not copy, translate, port, or lightly rewrite external implementation code.
- Do not copy external tests, fixtures, or code examples into this repository.
- Do not use code generators or imported project templates for new first-party code.
- Call existing dependencies through their public APIs when the design permits this use.
- Record each new dependency and its role in the design.
- Keep reference notes separate from implementation instructions.
- Do not describe this process as a clean-room process.

An installed authoring skill is a development tool.
Its installation does not make its source part of ApxInf.

## Design and interfaces

Read [the serving design](doc/serving-system-design-20261009.md) before serving work.
Read [the contract](doc/serving/contracts-v0.1.md) before an interface change.
Read [the implementation plan](doc/serving/implementation-plan.md) before a new stage.

The contract defines serving terms, fields, states, and errors.
The contract takes precedence over interface sketches in the design.
The existing v1 protocols retain their existing behavior.

- Define each interface before its implementation.
- Keep one term for each concept.
- Write Rust and Python types from the same contract.
- Check both implementations with the same original fixture corpus.
- Change the contract, types, validators, and fixtures together.
- Keep capability limits explicit.
- Do not silently change a provider, precision profile, or model revision.
- Complete each stage gate before dependent work.

## Technical writing

Use ASD-STE100 Issue 9 principles for normative English documents.
Use the installed `asd-ste100` skill when it is available.
Use active voice and short sentences.
Define each technical term once.
Keep instructions within 20 words.
Keep descriptions within 25 words.
Keep each paragraph to one topic and six sentences or fewer.

Keep Chinese explanations for discussion with the user.
Do not describe Chinese prose as compliant STE English.
Report structural checks separately from full dictionary review.
Do not claim certification from a skill or a linter.
