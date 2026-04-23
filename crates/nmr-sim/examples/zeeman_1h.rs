//! Demo: end-to-end 1H spectrum from the quantum engine for a non-interacting
//! three-spin system at 600 MHz.
//!
//! This is the M2a validation example. It wires together every piece of the
//! new quantum pipeline:
//!
//!   SpinSystem → ZeemanH → DiagonalPropagator → compute_fid → DiscreteSpectrum
//!
//! Because we only turn on the rotating-frame Zeeman term (no J-coupling yet),
//! the resulting spectrum is three singlets. In the next milestone we'll
//! turn on `JCouplingH` and the same pipeline — without any change below —
//! will produce the correct multiplets.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example zeeman_1h
//! ```
//!
//! Writes `zeeman_1h_spectrum.csv` in the current working directory.

use nmr_sim::operator::total_m_minus;
use nmr_sim::{
    apodize_exponential, compute_fid, thermal_x_state, zero_fill, DiagonalPropagator,
    DiscreteSpectrum, Isotope, Spin, SpinSystem, ZeemanH,
};

fn main() {
    // --- 1. Spin system ---
    let b0 = 14.0954; // Tesla → 600 MHz for 1H
    let sys = SpinSystem::new(
        vec![
            Spin::labeled(Isotope::H1, 1.5, "methyl"),
            Spin::labeled(Isotope::H1, 3.7, "methine"),
            Spin::labeled(Isotope::H1, 7.2, "aromatic"),
        ],
        b0,
    );
    println!("System: {} spins, Hilbert dim = {}", sys.len(), sys.dim());

    // --- 2. Hamiltonian ---
    let h = ZeemanH::new(&sys);

    // --- 3. Propagator ---
    //
    // Pick Δt so that 1/(2 Δt) comfortably exceeds the largest chemical-shift
    // offset we expect (≤ ~15 ppm → ~9 kHz at 600 MHz). 20 kHz sampling is
    // conservative and gives us a clean Nyquist margin.
    let dt = 1.0 / 20_000.0;
    let p = DiagonalPropagator::new(&h, dt);

    // --- 4. FID ---
    //
    // N = 65_536 × 50 μs = 3.28 s of "acquisition time". Raw frequency
    // resolution 1/(N·dt) ≈ 0.305 Hz ≈ 5×10⁻⁴ ppm at 600 MHz.
    //
    // Why this many points: with LB = 1 Hz (see step 5) the FID envelope
    // at t = N·dt is exp(−π·1·3.28) ≈ 3.5×10⁻⁵, so the rectangular window
    // imposed by truncation multiplies essentially zero — no boxcar-sinc
    // artifacts on the spectrum. A shorter N (8k, 16k, …) leaves meaningful
    // residual amplitude at the cutoff and produces visible sinc ripples
    // around each peak in absorption mode. On a 3-spin diagonal propagator
    // this is cheap; each FID sample is a 2^N_spins × 2^N_spins matrix.
    let n_points = 65_536;
    let rho0 = thermal_x_state(&sys);
    let observable = total_m_minus(&sys);
    let mut fid = compute_fid(&p, &rho0, &observable, n_points);

    // --- 5. Processing + spectrum ---
    //
    // Apodize with 1 Hz exponential line broadening (phenomenological stand-in
    // for T₂ until the relaxation milestone lands) and zero-fill 4× to
    // interpolate the peaks onto a smooth display grid.
    apodize_exponential(&mut fid, dt, 1.0);
    let fid = zero_fill(fid, 4);
    let spectrum = DiscreteSpectrum::from_fid(&fid, dt);
    println!(
        "Spectrum: {} bins, Δf = {:.2} Hz ({:.4} ppm)",
        spectrum.len(),
        spectrum.freq_resolution_hz(),
        spectrum.freq_resolution_hz() * 2.0 * std::f64::consts::PI * 1e6
            / (Isotope::H1.gamma() * b0),
    );

    // Report the three loudest peaks to stdout.
    let ppms = spectrum.frequencies_ppm(Isotope::H1, b0);
    let mags = spectrum.magnitude();
    let mut indexed: Vec<(usize, f64)> = mags.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top 3 peaks:");
    for (i, (idx, mag)) in indexed.iter().take(3).enumerate() {
        println!("  #{}: {:.3} ppm, mag = {:.3e}", i + 1, ppms[*idx], mag);
    }

    // --- 6. CSV output ---
    match spectrum.to_csv_ppm("zeeman_1h_spectrum.csv", Isotope::H1, b0) {
        Ok(()) => println!("Saved zeeman_1h_spectrum.csv"),
        Err(e) => eprintln!("Error writing CSV: {e}"),
    }
}
