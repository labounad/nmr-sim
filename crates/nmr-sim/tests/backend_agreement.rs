//! Cross-backend agreement: the quantum engine must match the first-order
//! engine wherever both apply.
//!
//! "Wherever both apply" = every spin is a singlet (no J-coupling, no strong
//! coupling). For singlets-only, both backends should produce a spectrum
//! with peaks at the same chemical shifts and relative heights equal to the
//! proton counts.
//!
//! This is an integration test rather than a unit test because it exercises
//! the public API end-to-end across several modules.

use nmr_sim::operator::total_m_minus;
use nmr_sim::{
    compute_fid, thermal_x_state, Coupling, DiagonalPropagator, DiscreteSpectrum, Hamiltonian,
    Isotope, JCouplingH, MatrixPropagator, Peak, PeakDef, Spectrum, Spin, SpinSystem, SumH,
    ZeemanH,
};

/// Find the global maximum of `values` within `±window` of a target x-value.
/// Returns `(x_at_max, y_at_max)`. Panics if no point is within the window.
fn max_within<'a>(xs: &'a [f64], ys: &'a [f64], target: f64, window: f64) -> (f64, f64) {
    let mut best: Option<(f64, f64)> = None;
    for (x, y) in xs.iter().zip(ys.iter()) {
        if (x - target).abs() > window {
            continue;
        }
        match best {
            None => best = Some((*x, *y)),
            Some((_, by)) if *y > by => best = Some((*x, *y)),
            _ => {}
        }
    }
    best.unwrap_or_else(|| panic!("no point within {window} ppm of {target}"))
}

#[test]
fn quantum_and_first_order_agree_for_singlet_only_three_site_system() {
    // Setup: three singlets at 1.0, 4.0, 7.5 ppm with proton counts 3, 1, 2.
    // Represented differently in each backend:
    //   - first-order: three PeakDefs with area = [3, 1, 2] and no couplings
    //   - quantum: a 6-spin system with three spins at 1.0 ppm, one at 4.0,
    //     two at 7.5, no J-coupling (handled by ZeemanH alone).

    let b0_tesla = 14.0954;
    let spectrometer_mhz = 600.0_f64; // matches 14.0954 T for 1H
    let shifts = [1.0, 4.0, 7.5];
    let counts = [3, 1, 2];
    let linewidth = 1.0;

    // ---- First-order backend ----
    let first_order_defs: Vec<PeakDef> = shifts
        .iter()
        .zip(counts.iter())
        .map(|(&shift, &count)| PeakDef {
            shift,
            area: count as f64,
            linewidth,
            couplings: Vec::<Coupling>::new(),
        })
        .collect();
    let first_order_peaks: Vec<Peak> = first_order_defs
        .iter()
        .flat_map(|d| d.expand(spectrometer_mhz))
        .collect();
    let fo_spectrum =
        Spectrum::new(first_order_peaks, 0.0, 10.0, spectrometer_mhz).with_num_points(20_000);
    let fo_data = fo_spectrum.compute();
    // Convert the Hz x-axis to ppm in-place for an apples-to-apples comparison.
    let fo_ppm: Vec<f64> = fo_data
        .iter()
        .map(|(f_hz, _)| f_hz / spectrometer_mhz)
        .collect();
    let fo_intensity: Vec<f64> = fo_data.iter().map(|(_, y)| *y).collect();

    // ---- Quantum backend ----
    let mut spins = Vec::new();
    for (&shift, &count) in shifts.iter().zip(counts.iter()) {
        for _ in 0..count {
            spins.push(Spin::new(Isotope::H1, shift));
        }
    }
    let sys = SpinSystem::new(spins, b0_tesla);
    let h = ZeemanH::new(&sys);
    let dt = 1.0 / 20_000.0;
    let n = 16_384;
    let p = DiagonalPropagator::new(&h, dt);
    let rho0 = thermal_x_state(&sys);
    let obs = total_m_minus(&sys);
    let fid = compute_fid(&p, &rho0, &obs, n);
    let qm_spectrum = DiscreteSpectrum::from_fid(&fid, dt);
    let qm_ppm = qm_spectrum.frequencies_ppm(Isotope::H1, b0_tesla);
    let qm_intensity = qm_spectrum.magnitude();

    // ---- Compare peak positions ----
    for &expected_shift in &shifts {
        let (fo_x, _) = max_within(&fo_ppm, &fo_intensity, expected_shift, 0.2);
        let (qm_x, _) = max_within(&qm_ppm, &qm_intensity, expected_shift, 0.2);
        assert!(
            (fo_x - expected_shift).abs() < 0.05,
            "first-order peak landed at {fo_x} (expected {expected_shift})",
        );
        assert!(
            (qm_x - expected_shift).abs() < 0.05,
            "quantum peak landed at {qm_x} (expected {expected_shift})",
        );
    }

    // ---- Compare relative intensities via per-peak L2 norm ----
    //
    // Why L2 rather than peak height? Peak *heights* are fragile across
    // backends:
    //   - The first-order Lorentzian is analytic, so its max is exact.
    //   - The quantum FFT is unapodized, so each peak is sinc-shaped. If a
    //     peak's true frequency sits at a non-integer bin (fractional offset
    //     Δ), the sampled max is attenuated by `sinc(Δ) = sin(πΔ)/(πΔ)`,
    //     which varies with Δ. Different peaks in the same spectrum get
    //     different fractional offsets → different attenuation → normalized
    //     peak-height ratios drift away from the true count ratios by a few
    //     percent. (This is what a commercial NMR simulator masks by
    //     applying an exponential window to the FID — we'd need that plus a
    //     matching Lorentzian convolution to use peak-height directly.)
    //
    // L2 norm over a peak window is by contrast *Parseval-exact* (up to the
    // tiny fraction of sinc-tail energy that leaks outside the window, which
    // is identical for all peaks of the same amplitude ratio):
    //   - First-order Lorentzian with area a over a window ≫ linewidth:
    //     ∫|L|² df ≈ a² / (2π·linewidth), so L2 ∝ a.
    //   - Quantum complex-exponential with amplitude A in an N-point FFT:
    //     window-integrated L2² ≈ A²·N² (Parseval), so L2 ∝ A.
    // Both scale linearly with the spin count, so normalized L2 ratios
    // recover the count ratios (3 : 1 : 2) regardless of lineshape.
    let l2_of = |ppms: &[f64], ints: &[f64]| -> Vec<f64> {
        shifts
            .iter()
            .map(|&s| {
                let sum_sq: f64 = ppms
                    .iter()
                    .zip(ints.iter())
                    .filter(|(p, _)| (**p - s).abs() < 0.15)
                    .map(|(_, y)| y * y)
                    .sum();
                sum_sq.sqrt()
            })
            .collect()
    };
    let fo_energies = l2_of(&fo_ppm, &fo_intensity);
    let qm_energies = l2_of(&qm_ppm, &qm_intensity);

    // Normalize each backend so the tallest peak's L2 = 1 and compare.
    let fo_max = fo_energies.iter().cloned().fold(0.0_f64, f64::max);
    let qm_max = qm_energies.iter().cloned().fold(0.0_f64, f64::max);
    for (i, (&fo_e, &qm_e)) in fo_energies.iter().zip(qm_energies.iter()).enumerate() {
        let fo_norm = fo_e / fo_max;
        let qm_norm = qm_e / qm_max;
        // 2% tolerance. Sources of residual disagreement:
        //   - Sinc tails extending past the ±0.15 ppm window (sub-percent).
        //   - Lorentzian tails extending past the window (sub-percent for
        //     linewidth ≪ window).
        //   - Finite-N Riemann-sum error in the L2 integral (sub-percent).
        assert!(
            (fo_norm - qm_norm).abs() < 0.02,
            "relative intensities disagree at {}: fo={fo_norm:.3}, qm={qm_norm:.3}",
            shifts[i],
        );
    }
}

#[test]
fn weak_coupling_limit_two_spins_matches_first_order_doublets() {
    // The quantum engine's strong-coupling J term (Îᵢ·Îⱼ) must reduce to the
    // first-order doublet pattern in the limit |J| ≪ |Δν|. That pattern is
    // two equal-intensity doublets:
    //
    //     A peaks:  ν_A ± J/2        B peaks:  ν_B ± J/2
    //     intensity: 1 : 1 : 1 : 1   (no roofing, by definition of weak coupling)
    //
    // We:
    //  (1) run the full quantum pipeline (ZeemanH + JCouplingH via SumH,
    //      routed through MatrixPropagator because the J term is non-diagonal),
    //  (2) check its four peak positions against the analytical ν ± J/2 form,
    //  (3) compare to the existing first-order backend (PeakDef with `Coupling`
    //      of n_neighbors=1) as a second cross-check.
    //
    // System: two protons at 7.0 and 1.0 ppm (|Δν| = 3600 Hz at 600 MHz),
    // coupled by J = 15 Hz. J / |Δν| ≈ 0.004 — deeply first-order.
    let b0_tesla = 14.0954;
    let spectrometer_mhz = 600.0_f64;
    let shift_a = 7.0;
    let shift_b = 1.0;
    let j_hz = 15.0;
    let linewidth = 1.0;

    // Expected peak positions in ppm. J (Hz) → ppm via divide by
    // spectrometer_mhz (because 1 ppm = spectrometer_mhz Hz for that isotope).
    let half_j_ppm = j_hz / 2.0 / spectrometer_mhz;
    let expected_a_hi = shift_a + half_j_ppm;
    let expected_a_lo = shift_a - half_j_ppm;
    let expected_b_hi = shift_b + half_j_ppm;
    let expected_b_lo = shift_b - half_j_ppm;
    let expected_peaks = [expected_a_hi, expected_a_lo, expected_b_hi, expected_b_lo];

    // ---- Quantum backend ----
    //
    // Sampling: need ≥ ~1 Hz frequency resolution to cleanly separate
    // multiplet lines 15 Hz apart. dt = 1/20_000, n = 32_768 → Δf ≈ 0.61 Hz,
    // total acquisition ≈ 1.64 s. Comfortable resolution budget.
    let sys = SpinSystem::new(
        vec![
            Spin::new(Isotope::H1, shift_a),
            Spin::new(Isotope::H1, shift_b),
        ],
        b0_tesla,
    );
    let h: Box<dyn Hamiltonian> = Box::new(SumH::new(vec![
        Box::new(ZeemanH::new(&sys)),
        Box::new(JCouplingH::new(&sys, &[(0, 1, j_hz)])),
    ]));
    // SumH containing a JCouplingH is non-diagonal; DiagonalPropagator would
    // panic. MatrixPropagator is the correct universal fallback.
    assert!(
        h.try_as_diagonal().is_none(),
        "sanity: H_Z + H_J must not advertise diagonal storage",
    );
    let dt = 1.0 / 20_000.0;
    let n = 32_768;
    let p = MatrixPropagator::new(h.as_ref(), dt);
    let rho0 = thermal_x_state(&sys);
    let obs = total_m_minus(&sys);
    let fid = compute_fid(&p, &rho0, &obs, n);
    let qm_spectrum = DiscreteSpectrum::from_fid(&fid, dt);
    let qm_ppm = qm_spectrum.frequencies_ppm(Isotope::H1, b0_tesla);
    let qm_intensity = qm_spectrum.magnitude();

    // (2) Check quantum peak positions against the analytical prediction.
    // Use a half-window of 0.5 × J/(spectrometer_mhz) = ½ the multiplet
    // splitting, so each peak's window is disjoint from its multiplet
    // sibling's.
    let window = half_j_ppm;
    let tol_ppm = 0.003; // ~2 Hz at 600 MHz — finer than one FFT bin (0.61 Hz)
    for &target in &expected_peaks {
        let (qm_x, _) = max_within(&qm_ppm, &qm_intensity, target, window);
        assert!(
            (qm_x - target).abs() < tol_ppm,
            "quantum peak at {qm_x:.5} ppm, expected {target:.5} ppm (tol {tol_ppm})",
        );
    }

    // (3) Sanity-check the first-order backend at the same setup: two
    // PeakDefs, each with a single neighbor coupling of J Hz → doublet each.
    let fo_defs = [
        PeakDef {
            shift: shift_a,
            area: 1.0,
            linewidth,
            couplings: vec![Coupling {
                j_hz,
                n_neighbors: 1,
            }],
        },
        PeakDef {
            shift: shift_b,
            area: 1.0,
            linewidth,
            couplings: vec![Coupling {
                j_hz,
                n_neighbors: 1,
            }],
        },
    ];
    let fo_peaks: Vec<Peak> = fo_defs
        .iter()
        .flat_map(|d| d.expand(spectrometer_mhz))
        .collect();
    let fo_spectrum = Spectrum::new(fo_peaks, 0.0, 10.0, spectrometer_mhz).with_num_points(50_000);
    let fo_data = fo_spectrum.compute();
    let fo_ppm: Vec<f64> = fo_data
        .iter()
        .map(|(f_hz, _)| f_hz / spectrometer_mhz)
        .collect();
    let fo_intensity: Vec<f64> = fo_data.iter().map(|(_, y)| *y).collect();

    for &target in &expected_peaks {
        let (fo_x, _) = max_within(&fo_ppm, &fo_intensity, target, window);
        assert!(
            (fo_x - target).abs() < tol_ppm,
            "first-order peak at {fo_x:.5} ppm, expected {target:.5} ppm (tol {tol_ppm})",
        );
    }

    // (4) Compare relative intensities per peak via L2 norm, same Parseval
    // argument as in the singlets test above. In the weak-coupling limit all
    // four multiplet lines are equal-intensity, so normalized L2 should be
    // within a few percent of uniform (1, 1, 1, 1).
    let l2_of = |ppms: &[f64], ints: &[f64]| -> Vec<f64> {
        expected_peaks
            .iter()
            .map(|&target| {
                let sum_sq: f64 = ppms
                    .iter()
                    .zip(ints.iter())
                    .filter(|(p, _)| (**p - target).abs() < window)
                    .map(|(_, y)| y * y)
                    .sum();
                sum_sq.sqrt()
            })
            .collect()
    };
    let qm_l2 = l2_of(&qm_ppm, &qm_intensity);
    let fo_l2 = l2_of(&fo_ppm, &fo_intensity);
    let qm_max = qm_l2.iter().cloned().fold(0.0_f64, f64::max);
    let fo_max = fo_l2.iter().cloned().fold(0.0_f64, f64::max);
    for i in 0..expected_peaks.len() {
        let qm_norm = qm_l2[i] / qm_max;
        let fo_norm = fo_l2[i] / fo_max;
        // 3% tolerance: slightly looser than the singlet test because the
        // quantum peaks here are much closer together (≈ 15 Hz apart vs the
        // hundreds of Hz separations in the singlets case), so sinc-tail leakage
        // between multiplet siblings — and between A and B doublets — is larger.
        assert!(
            (qm_norm - 1.0).abs() < 0.03,
            "quantum multiplet line {i} at {:.4} ppm has normalized L2 {qm_norm:.3}, \
             expected ~1.0 (weak-coupling limit → equal intensities)",
            expected_peaks[i],
        );
        assert!(
            (fo_norm - 1.0).abs() < 0.03,
            "first-order multiplet line {i} at {:.4} ppm has normalized L2 {fo_norm:.3}, \
             expected ~1.0",
            expected_peaks[i],
        );
        assert!(
            (qm_norm - fo_norm).abs() < 0.03,
            "backends disagree on line {i}: qm={qm_norm:.3}, fo={fo_norm:.3}",
        );
    }
}
