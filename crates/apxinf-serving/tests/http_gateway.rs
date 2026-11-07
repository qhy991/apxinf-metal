//! Real HTTP checks with an original CPU protocol fixture.
//!
//! The Python fixture runs every case except the output-budget boundary, which
//! `http_output_budget.rs` selects into its own target.
#![cfg(unix)]

#[path = "support/python_fixture.rs"]
mod python_fixture;

#[test]
fn real_gateway_with_cpu_worker() {
    python_fixture::run(&[
        "--exclude",
        "OutputBudgetTests",
        "--expect-cases",
        "15",
        "--expect-total",
        "17",
    ]);
}
