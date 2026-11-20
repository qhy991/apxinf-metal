//! The checkpoint's `calibration.json`: temperature scaling of option scores
//! and monotone tables for the typed answers' probabilities
//! (`calibration.py`). Pure host math.

use super::questions::Question;
use apxinf_core::{Error, Result};

fn err(message: impl Into<String>) -> Error {
    Error::Other(format!("d1 calibration: {}", message.into()))
}

#[derive(Clone, Debug)]
pub struct Calibration {
    pub temperatures: std::collections::HashMap<String, f64>,
    pub default: f64,
    pub tables: std::collections::HashMap<String, Vec<(f64, f64)>>,
}

/// The reference's `question_bucket`.
fn bucket(q: &Question, cardinality: usize) -> String {
    match q {
        Question::Noul { .. } => "noul".into(),
        Question::Score { .. } => format!("score:{cardinality}"),
        _ => {
            if cardinality <= 3 {
                "choice:small".into()
            } else if cardinality <= 8 {
                "choice:mid".into()
            } else if cardinality <= 26 {
                "choice:large".into()
            } else {
                "choice:inventory".into()
            }
        }
    }
}

/// `calibration.through`: `value` through increasing (raw, calibrated) knots,
/// linear between them, flat past the ends.
pub fn through(value: f64, knots: &[(f64, f64)]) -> f64 {
    if knots.is_empty() {
        return value;
    }
    // `bisect_left` on the raw column: first index whose raw >= value.
    let idx = knots.partition_point(|(raw, _)| *raw < value);
    if idx == 0 {
        return knots[0].1;
    }
    if idx >= knots.len() {
        return knots[knots.len() - 1].1;
    }
    let (a, pa) = knots[idx - 1];
    let (b, pb) = knots[idx];
    pa + (value - a) * (pb - pa) / (b - a)
}

impl Calibration {
    pub fn from_json(text: &str) -> Result<Self> {
        let json: serde_json::Value =
            serde_json::from_str(text).map_err(|e| err(format!("invalid calibration.json: {e}")))?;
        let default = json
            .get("default")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let temperatures = json
            .get("temperatures")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_f64().map(|x| (k.clone(), x)))
                    .collect()
            })
            .unwrap_or_default();
        let tables = json
            .get("tables")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| {
                        v.as_array().map(|rows| {
                            (
                                k.clone(),
                                rows.iter()
                                    .filter_map(|row| {
                                        row.as_array().and_then(|pair| {
                                            Some((pair.first()?.as_f64()?, pair.get(1)?.as_f64()?))
                                        })
                                    })
                                    .collect(),
                            )
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            temperatures,
            default,
            tables,
        })
    }

    pub fn temperature(&self, q: &Question, cardinality: usize) -> f64 {
        self.temperatures
            .get(&bucket(q, cardinality))
            .copied()
            .unwrap_or(self.default)
    }

    pub fn through(&self, name: &str, value: f64) -> f64 {
        through(
            value,
            self.tables.get(name).map(Vec::as_slice).unwrap_or(&[]),
        )
    }

    /// `Calibration.level`: the level whose observed share is `share` — the
    /// table read backwards (both columns increase).
    pub fn level(&self, name: &str, share: f64) -> f64 {
        let inverse: Vec<(f64, f64)> = self
            .tables
            .get(name)
            .map(|rows| rows.iter().map(|(x, y)| (*y, *x)).collect())
            .unwrap_or_default();
        through(share, &inverse)
    }
}
