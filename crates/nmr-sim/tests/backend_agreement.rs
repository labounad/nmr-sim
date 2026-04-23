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
    compute_fid, thermal_x_state, Coupling, DiagonalPropagator, DiscreteSpectrum, Isotope, Peak,
    PeakDef, Spectrum, Spin, SpinSystem, ZeemanH,
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
