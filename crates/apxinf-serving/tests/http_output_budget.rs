//! Real HTTP checks for the public output-budget contract.
//!
//! The gateway keeps an omitted `max_tokens` inside the deployment output
//! budget and rejects an explicit zero before admission. The Python fixture
//! owns the fixture launch, lifecycle, and cleanup; these targets only select
//! its output-budget cases so `cargo test` reports them separately.
#![cfg(unix)]

#[path = "support/python_fixture.rs"]
mod python_fixture;

#[test]
fn omitted_limit_is_admitted_with_the_deployment_budget() {
    python_fixture::run(&[
        "--select",
        "OutputBudgetTests.test_omitted_limit_is_admitted_with_the_deployment_budget",
        "--expect-cases",
        "1",
        "--expect-total",
        "18",
    ]);
}

#[test]
fn every_output_budget_case_has_a_target() {
    // The two targets select methods individually, so a new class case would
    // otherwise never run under `cargo test`.
    python_fixture::run(&[
        "--select",
        "OutputBudgetTests",
        "--expect-cases",
        "2",
        "--count-only",
    ]);
}

#[test]
fn exhausted_output_budget_rejects_generation_but_counts_tokens() {
    python_fixture::run(&[
        "--select",
        "OutputBudgetTests.test_exhausted_output_budget_rejects_generation_but_counts_tokens",
        "--expect-cases",
        "1",
        "--expect-total",
        "18",
    ]);
}
