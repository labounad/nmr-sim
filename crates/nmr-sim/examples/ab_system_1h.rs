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
//! Writes `ab_1h_<mhz>mhz_spectrum.csv` once per field in the working
//! directory. Inspect any of them with `scripts/plot_spectrum.py`.

use nmr_sim::operator::total_m_minus;
use nmr_sim::{
    compute_fid, thermal_x_state, DiscreteSpectrum, Hamiltonian, Isotope, JCouplingH,
    MatrixPropagator, Spin, SpinSystem, SumH, ZeemanH,
};

/// Chemical shifts (ppm) — fixed across the field scan.
const SHIFT_A_PPM: f64 = 3.0;
const SHIFT_B_PPM: f64 = 3.5;
/// Scalar coupling (Hz) — fixed across the field scan.
const J_HZ: f64 = 7.0;

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
    // Longer N = sharper multiplet lines. J = 7 Hz needs ≲ 1 Hz resolution
    // to visibly resolve the splitting → N·Δt ≳ 1 s → N ≳ 20_000.
    // 16_384 points gives Δf ≈ 1.22 Hz, which is good enough for roofing
    // patterns to be obvious by eye.
    let rho0 = thermal_x_state(&sys);
    let obs = total_m_minus(&sys);
    let n_points = 16_384;
    let fid = compute_fid(&p, &rho0, &obs, n_points);

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

    // Save to CSV, one file per field, in the working directory.
    let filename = format!("ab_1h_{}mhz_spectrum.csv", mhz as u32);
    match spectrum.to_csv_ppm(&filename, Isotope::H1, b0) {
        Ok(()) => println!("Saved {filename}"),
        Err(e) => eprintln!("Error writing {filename}: {e}"),
    }
}

fn main() {
    println!(
        "AB two-proton system: δ_A = {SHIFT_A_PPM} ppm, δ_B = {SHIFT_B_PPM} ppm, \
         J = {J_HZ} Hz. Scanning B₀ to walk strong → weak coupling."
    );

    // From low field (strong coupling dominant) to high field (first-order
    // doublets emerge). 60 MHz is "bench-top" NMR; 1000 MHz is the current
    // state of the art.
    for mhz in [60.0, 200.0, 600.0, 1000.0] {
        simulate_at(mhz);
    }

    println!(
        "\nPlot any file with:\n    python scripts/plot_spectrum.py ab_1h_60mhz_spectrum.csv\n",
    );
}
