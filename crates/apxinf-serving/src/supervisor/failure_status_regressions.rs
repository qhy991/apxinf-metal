//! Worker-declared failure codes keep the documented HTTP status.
//!
//! `doc/serving/contracts-v0.1.md` ("Faults and recovery") fixes the default
//! public status for every error code. The supervisor is the only place that
//! turns a worker error record into a public result, so this table is the
//! contract lock for that mapping.
use super::*;
use crate::contracts::ERRORS;

/// The documented default status before stream headers.
fn contract_status(code: &str) -> u16 {
    match code {
        "invalid_request" | "unsupported_feature" | "context_limit" => 400,
        "model_not_found" | "session_not_found" => 404,
        "session_conflict" | "prefix_mismatch" => 409,
        "queue_full" | "quota_exceeded" => 429,
        "capacity_unavailable" | "model_unavailable" | "worker_lost" => 503,
        "deadline_exceeded" => 504,
        "protocol_fault" | "internal_error" | "state_invalid" => 500,
        other => panic!("The contract table has no status for {other}."),
    }
}

#[test]
fn every_worker_failure_code_keeps_its_contract_status() {
    for code in ERRORS {
        let frame = json!({"error": {
            "code": code,
            "message": "The worker declared this failure.",
            "scope": "request",
            "state_validity": "none",
        }});
        let error = error_from_frame(&frame);
        assert_eq!(
            (error.status, error.code.as_str()),
            (contract_status(code), *code),
            "{code}"
        );
        assert_eq!(error.message, "The worker declared this failure.", "{code}");
    }
}
