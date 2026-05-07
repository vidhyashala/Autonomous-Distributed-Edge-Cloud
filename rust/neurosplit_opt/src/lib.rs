//! Rust optimization primitives for NeuroSplit-X.

/// Scores a candidate split with latency, energy, bandwidth, privacy, and stability terms.
pub fn score_split(latency_ms: f64, energy_j: f64, bandwidth_mb: f64, privacy_risk: f64, instability: f64) -> f64 {
    0.38 * latency_ms + 2.4 * energy_j + 0.16 * bandwidth_mb + 14.0 * privacy_risk + 1.6 * instability
}

/// Returns the index of the lowest-cost candidate.
pub fn argmin(scores: &[f64]) -> Option<usize> {
    scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmin_selects_lowest_score() {
        assert_eq!(argmin(&[3.0, 1.0, 2.0]), Some(1));
    }
}
