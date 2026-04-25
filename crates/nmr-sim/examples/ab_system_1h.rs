//! Demo: the strong → weak coupling transition in a two-proton AB system.
//!
//! This is the M3 validation example. It wires together the new J-coupling
//! machinery:
//!
//!   SpinSystem → ZeemanH + JCouplingH → SumH → MatrixPropagator
//!              → compute_fid → DiscreteSpectrum
//!
//! # Why this example matters
//!
//! Two protons at shifts δ_A, δ_B (ppm) coupled by J (Hz) produce a spectrum
//! that depends on the dimensionless ratio
//!
//!     ρ = J / |ν_A − ν_B| = J / (|δ_A − δ_B| · ν_spectrometer)
//!
//! When ρ ≪ 1 (weak coupling, high field) you see two equal-intensity
//! doublets. As ρ grows the inner lines rise and the outer lines shrink
//! ("roofing"); at ρ ~ 1 you get a classic strongly-coupled AB quartet;
//! at ρ → ∞ the four lines collapse to a singlet at the chemical-shift
//! average.
//!
//! A first-order simulator can *only* produce the ρ → 0 limit. The quantum
//! engine produces the full spectrum for any ρ. This example scans B₀ at
//! fixed chemical shifts and fixed J to walk across the transition, because
//! that's the knob real spectroscopists actually turn: buy a stronger magnet
//! and your J / |Δν| gets smaller.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example ab_system_1h
//! ```
//!
//! Writes `examples/outputs/ab_1h_<mhz>mhz_spectrum.csv` once per field
//! (directory created if absent). Inspect any of them with
//! `scripts/plot_spectrum.py`.

use std::fs::create_dir_all;

use nmr_sim::operator::total_m_minus;
use nmr_sim::{
    apodize_exponential, compute_fid, thermal_x_state, zero_fill, DiscreteSpectrum, Hamiltonian,
    Isotope, JCouplingH, MatrixPropagator, Spin, SpinSystem, SumH, ZeemanH,
};

/// Centralised output directory shared by every example.
const OUTPUT_DIR: &str = "examples/outputs";

/// Chemical shifts (ppm) — fixed across the field scan.
const SHIFT_A_PPM: f64 = 3.0;
const SHIFT_B_PPM: f64 = 3.5;
/// Scalar coupling (Hz) — fixed across the field scan.
const J_HZ: f64 = 7.0;
/// Exponential line broadening applied to the FID before FFT (Hz). 1 Hz is
/// the conventional default for 1D 1H: large enough that each peak spans
/// several FFT bins after zero-filling, small enough to leave J-splittings
/// resolved. When real T₂ relaxation lands in a later milestone this should
/// drop to 0.
const LB_HZ: f64 = 1.0;
/// Zero-fill factor. Frequency-domain interpolation only — the underlying
/// resolution is still 1 / (N · dt). 4× matches Topspin's default for 1D.
const ZF_FACTOR: usize = 4;

fn b0_for_mhz(mhz: f64) -> f64 {
    // 14.0954 T ↔ 600 MHz for 1H; scale linearly.
    14.0954 * mhz / 600.0
}

fn simulate_at(mhz: f64) {
    let b0 = b0_for_mhz(mhz);

    // ---- Spin system ----
    let sys = SpinSystem::new(
        vec![
            Spin::labeled(Isotope::H1, SHIFT_A_PPM, "A"),
            Spin::labeled(Isotope::H1, SHIFT_B_PPM, "B"),
        ],
        b0,
    );

    // ---- Hamiltonian: H_Z + H_J ----
    //
    // SumH is the trait-level "addition" of Hamiltonian contributions. Adding
    // more physics (relaxation, exchange, dipolar, …) will look exactly like
    // this: build the new term's Hamiltonian, push it into the Vec.
    let h_zeeman = ZeemanH::new(&sys);
    let h_coupling = JCouplingH::new(&sys, &[(0, 1, J_HZ)]);
    let h: Box<dyn Hamiltonian> =
        Box::new(SumH::new(vec![Box::new(h_zeeman), Box::new(h_coupling)]));

    // ---- Propagator ----
    //
    // H_J carries flip-flop (off-diagonal) terms, so H overall is non-
    // diagonal. DiagonalPropagator would panic here; MatrixPropagator is the
    // universal fallback. Under the hood it does a single matrix exp of
    // (−iHΔt) via nalgebra's scaling-and-squaring Padé, and then each FID
    // step is a triple matmul `U ρ U†`.
    //
    // Δt chosen by a conservative Nyquist margin: spectral content lives
    // within ±~10 ppm at 1H = ±6 kHz at 600 MHz; 20 kHz sampling leaves room.
    let dt = 1.0 / 20_000.0;
    let p = MatrixPropagator::new(h.as_ref(), dt);

    // ---- FID ----
    //
    // Two concerns set the lower bound on N:
    //
    // 1. Resolving the J-splitting. J = 7 Hz needs ≲ 1 Hz bin width →
    //    N · Δt ≳ 1 s → N ≳ 20_000.
    // 2. Letting the FID decay before truncation. With LB = 1 Hz the
    //    envelope is exp(−π · LB · t); at t = N · Δt = 3.28 s (N = 65_536,
    //    Δt = 50 μs) that's ≈ 3.5 × 10⁻⁵. If we stop earlier the residual
    //    amplitude gets truncated by a rectangular window and the FFT
    //    picks up a boxcar-sinc — in absorption mode that shows up as
    //    signed ripples of period 1/T around each peak, 7%-of-peak at
    //    0.82 s of acquisition (N = 16_384) and falling to ~invisible by
    //    3.28 s (N = 65_536). This is the "buy a longer FID" trick that
    //    real spectrometers use implicitly via T₂ relaxation.
    //
    // 65_536 points also gives Δf_raw = 0.305 Hz → 0.076 Hz after 4× zero-
    // fill → ~13 bins per FWHM, so peaks render smooth instead of pointy.
    // Cost for an AB (4×4 matrix) system: 4× more matmul steps, still
    // sub-millisecond per field.
    let rho0 = thermal_x_state(&sys);
    let obs = total_m_minus(&sys);
    let n_points = 65_536;
    let mut fid = compute_fid(&p, &rho0, &obs, n_points);

    // ---- Processing: apodize then zero-fill ----
    //
    // Without apodization the truncated undamped FID would FFT to a sinc
    // (narrow main lobe + side-lobe ringing), and the raw bin width
    // Δf = 1/(N·dt) ≈ 1.22 Hz would put only ~1 bin across the main lobe —
    // so the rendered peaks look like spikes, not lines. Exponential
    // apodization convolves the spectrum with a Lorentzian of FWHM ≈ LB_HZ;
    // zero-filling interpolates that Lorentzian onto a finer bin grid.
    apodize_exponential(&mut fid, dt, LB_HZ);
    let fid = zero_fill(fid, ZF_FACTOR);

    // ---- Spectrum ----
    let spectrum = DiscreteSpectrum::from_fid(&fid, dt);

    // Report the top four peaks — that's one multiplet's worth for an AB.
    let ppms = spectrum.frequencies_ppm(Isotope::H1, b0);
    let mags = spectrum.magnitude();
    let mut indexed: Vec<(usize, f64)> = mags.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let delta_ppm = (SHIFT_A_PPM - SHIFT_B_PPM).abs();
    let delta_nu_hz = delta_ppm * mhz;
    let rho = J_HZ / delta_nu_hz;

    println!(
        "\n== {:>5.0} MHz ({:>7.4} T) — J/|Δν| = {J_HZ}/{:.1} = {:.4} ==",
        mhz, b0, delta_nu_hz, rho,
    );
    println!("Top 4 peaks:");
    for (rank, (idx, mag)) in indexed.iter().take(4).enumerate() {
        println!(
            "  #{}: {:7.4} ppm   mag = {:.3e}",
            rank + 1,
            ppms[*idx],
            mag
        );
    }

    // Save to CSV, one file per field, in the shared examples/outputs/.
    let path = format!("{OUTPUT_DIR}/ab_1h_{}mhz_spectrum.csv", mhz as u32);
    match spectrum.to_csv_ppm(&path, Isotope::H1, b0) {
        Ok(()) => println!("Saved {path}"),
        Err(e) => eprintln!("Error writing {path}: {e}"),
    }
}

fn main() {
    println!(
        "AB two-proton system: δ_A = {SHIFT_A_PPM} ppm, δ_B = {SHIFT_B_PPM} ppm, \
         J = {J_HZ} Hz. Scanning B₀ to walk strong → weak coupling."
    );

    create_dir_all(OUTPUT_DIR).expect("failed to create output directory");

    // From low field (strong coupling dominant) to high field (first-order
    // doublets emerge). 60 MHz is "bench-top" NMR; 1000 MHz is the current
    // state of the art.
    for mhz in [60.0, 200.0, 600.0, 1000.0] {
        simulate_at(mhz);
    }

    println!(
        "\nPlot any file with:\n    python scripts/plot_spectrum.py {OUTPUT_DIR}/ab_1h_60mhz_spectrum.csv\n",
    );
}
