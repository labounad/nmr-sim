#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coupling {
    pub j_hz: f64,
    pub n_neighbors: usize,
}

pub fn pascal_row(n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![1.0];
    }

    let mut row = vec![1.0];
    for i in 0..n {
        let mut new_row = vec![1.0];
        for j in 0..i {
            new_row.push(row[j] + row[j + 1]);
        }
        new_row.push(1.0);
        row = new_row;
    }

    row
}

// TODO: If equivalent neighbors are modeled as separate Coupling entries
// (e.g., three n_neighbors=1 instead of one n_neighbors=3), we get 2^n
// sub-peaks instead of n+1. Could add a merge step to deduplicate
// overlapping positions. Not urgent — the design encourages grouping
// equivalent neighbors into one Coupling, which avoids the blowup.

/// Expand a single peak into sub-peaks by applying J-couplings.
///
/// Each coupling splits every existing sub-peak into a multiplet
/// using Pascal's triangle intensities. Couplings are applied
/// sequentially, so a peak with two couplings gets split twice.
///
/// Returns a Vec of (shift_ppm, relative_area) tuples.
pub fn expand_peak(
    shift: f64,
    area: f64,
    couplings: &[Coupling],
    spectrometer_freq: f64,
) -> Vec<(f64, f64)> {
    let mut sub_peaks = vec![(shift, area)];

    for coupling in couplings {
        let coeffs = pascal_row(coupling.n_neighbors);
        let coeff_sum: f64 = coeffs.iter().sum();
        let j_ppm = coupling.j_hz / spectrometer_freq;
        let n = coupling.n_neighbors as f64;

        let mut new_sub_peaks = Vec::new();
        for (center, sub_area) in &sub_peaks {
            for (k, &coeff) in coeffs.iter().enumerate() {
                let offset = (k as f64 - n / 2.0) * j_ppm;
                new_sub_peaks.push((center + offset, sub_area * coeff / coeff_sum));
            }
        }
        sub_peaks = new_sub_peaks;
    }

    sub_peaks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pascal_row() {
        assert_eq!(pascal_row(0), vec![1.0]);
        assert_eq!(pascal_row(1), vec![1.0, 1.0]);
        assert_eq!(pascal_row(2), vec![1.0, 2.0, 1.0]);
        assert_eq!(pascal_row(3), vec![1.0, 3.0, 3.0, 1.0]);
    }

    #[test]
    fn test_expand_no_couplings() {
        // No couplings: peak comes back unchanged
        let result = expand_peak(1.0, 6.0, &[], 600.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (1.0, 6.0));
    }

    #[test]
    fn test_expand_doublet() {
        // 1 neighbor → doublet, two equal sub-peaks
        let couplings = vec![Coupling {
            j_hz: 6.0,
            n_neighbors: 1,
        }];
        let result = expand_peak(1.0, 2.0, &couplings, 600.0);
        assert_eq!(result.len(), 2);
        // Each sub-peak gets half the area
        assert!((result[0].1 - 1.0).abs() < 1e-10);
        assert!((result[1].1 - 1.0).abs() < 1e-10);
        // Separated by J = 6 Hz = 0.01 ppm at 600 MHz
        let separation = result[1].0 - result[0].0;
        assert!((separation - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_expand_triplet() {
        // 2 neighbors → triplet with 1:2:1 pattern
        let couplings = vec![Coupling {
            j_hz: 6.0,
            n_neighbors: 2,
        }];
        let result = expand_peak(5.0, 4.0, &couplings, 600.0);
        assert_eq!(result.len(), 3);
        // Areas should be 1:2:1 scaled to total area 4.0
        assert!((result[0].1 - 1.0).abs() < 1e-10);
        assert!((result[1].1 - 2.0).abs() < 1e-10);
        assert!((result[2].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_area_conserved() {
        // Total area should be preserved through any splitting
        let couplings = vec![
            Coupling {
                j_hz: 7.0,
                n_neighbors: 3,
            },
            Coupling {
                j_hz: 5.0,
                n_neighbors: 2,
            },
        ];
        let result = expand_peak(3.0, 10.0, &couplings, 600.0);
        let total_area: f64 = result.iter().map(|(_, a)| a).sum();
        assert!((total_area - 10.0).abs() < 1e-10);
    }
}
