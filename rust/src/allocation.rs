//! Long-only signal allocations above point-in-time base-weight floors.

use crate::matrix;

/// Allocate known signals by average rank above each stock's floor.
///
/// All members receive their base floor. The entire remaining allocation is
/// spread over known signals by positive direction-adjusted average ranks;
/// tied scores receive equal overlay weights. No known signals means the
/// complete base portfolio. Weights are long only.
pub fn target_weights(base_weights: &[f64], values: &[f64], keep: f64, direction: i64) -> Vec<f64> {
    let known: Vec<usize> = (0..base_weights.len())
        .filter(|&i| base_weights[i] > 0.0 && values[i].is_finite())
        .collect();
    if known.is_empty() || keep == 1.0 {
        return base_weights.to_vec();
    }
    let scores: Vec<f64> = known
        .iter()
        .map(|&i| values[i] * direction as f64)
        .collect();
    // np.unique: sorted distinct values, inverse indices and counts.
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    let mut ranks = vec![0.0; scores.len()];
    let mut start = 0;
    let mut end_cumulative = 0usize;
    while start < order.len() {
        let mut stop = start + 1;
        while stop < order.len() && scores[order[stop]] == scores[order[start]] {
            stop += 1;
        }
        let count = stop - start;
        end_cumulative += count;
        let rank = end_cumulative as f64 - (count as f64 - 1.0) / 2.0;
        for &position in &order[start..stop] {
            ranks[position] = rank;
        }
        start = stop;
    }
    let base_sum = matrix::sum(base_weights);
    let rank_sum = matrix::sum(&ranks);
    let mut weights: Vec<f64> = base_weights.iter().map(|w| keep * w).collect();
    for (k, &i) in known.iter().enumerate() {
        weights[i] += (1.0 - keep) * base_sum * ranks[k] / rank_sum;
    }
    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_and_ties() {
        let base = [0.25, 0.25, 0.25, 0.25];
        let values = [1.0, 2.0, 2.0, f64::NAN];
        let weights = target_weights(&base, &values, 0.8, 1);
        let total: f64 = weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!((weights[1] - weights[2]).abs() < 1e-15);
        assert!((weights[3] - 0.2).abs() < 1e-15);
        assert_eq!(target_weights(&base, &[f64::NAN; 4], 0.8, 1), base.to_vec());
    }
}
