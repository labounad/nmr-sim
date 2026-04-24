//! Demo: 1H spectrum of 5-methylcyclohex-2-en-1-one at 600 MHz.
//!
//! This is the first "real molecule" test of the quantum engine: 10 protons,
//! 8 chemically-distinct groups, 14 measured non-zero J-couplings (12 inter-
//! ring-group + 2 ring-to-methyl). It is also the backend-stress test that
//! motivated [`EigenPropagator`]: `MatrixPropagator` at D = 2¹⁰ = 1024 would
//! spend hours on the FID loop, while the eigendecomposition route diagonal-
//! izes H once (O(D³)) and then runs the entire N-point FID at O(N·D²).
//!
//! # Spin system
//!
//! Labels A..H come directly from Lucas's experimentally-derived assignment
//! spreadsheet. A..G are distinct ring / methine protons; H is the methyl
//! triplet (3 equivalent protons). We model the methyl as three independent
//! spin-1/2's at the same chemical shift (1.07 ppm) with zero intra-methyl
//! coupling — equivalent protons don't split each other in their own
//! spectrum, and this "explicit 3× copy" encoding is correct for the
//! observable first-order behavior without needing symmetry-reduction
//! machinery (that's a future optimization for the 12+ spin regime).
//!
//! Effective dimension: 2^10 = 1024 Hilbert states.
//!
//! # J-coupling topology
//!
//! Twelve inter-group couplings among the seven A..G ring / methine spins,
//! plus two ring-to-methyl couplings (C-H=1.0, E-H=6.6) that are replicated
//! across the three methyl protons → 12 + 2·3 = 18 pairs in total. See
//! `GROUP_COUPLINGS` and `METHYL_COUPLINGS` for the full lists. All
//! magnitudes only (signs unknown from experiment, and the quantum engine
//! doesn't need them — the roofing pattern / line positions only depend on
//! |J|).
//!
//! Key splittings to watch:
//! - A-B = 10.1 Hz: endocyclic vinyl doublet-of-doublets (with A-D and A-G
//!   in a complex allylic pattern).
//! - C-F = 16.1 Hz and D-G = 18.6 Hz: geminal CH₂ couplings — big and
//!   unambiguous.
//! - E-F = 12.4 Hz: the methine-to-geminal-proton coupling.
//! - E-H = 6.6 Hz: the methyl triplet / methine-to-methyl coupling.
//!
//! # Strong vs weak coupling
//!
//! At 600 MHz, |Δν| between the most closely-spaced pairs (e.g. C at 2.48
//! vs D at 2.42, Δδ = 0.06 ppm → Δν = 36 Hz) dominates over J (max 18.6 Hz),
//! so we're weakly coupled (J/|Δν| ≲ 0.5 for the worst pair, mostly ≲ 0.1).
//! The first-order backend would get line *positions* right but not the
//! roofing / intensity asymmetries. The quantum engine gets both.
//!
//! # Runtime expectations
//!
//! - Eigendecomposition of a 1024×1024 real-symmetric matrix: a few seconds
//!   on the pure-Rust path, under a second with threaded LAPACK.
//! - FID at N = 65,536: ~1–3 minutes depending on BLAS backend.
//! - Total wall-clock: a handful of minutes on a laptop.
//!
//! Run with:
//!
//! ```sh
//! # Pure-Rust default (portable, single-threaded eigensolve)
//! cargo run --release --example five_me_cyclohexenone_1h
//!
//! # Threaded LAPACK (Accelerate on macOS / Apple Silicon)
//! cargo run --release --features lapack-accelerate \
//!     --example five_me_cyclohexenone_1h
//!
//! # Threaded LAPACK (OpenBLAS on Linux)
//! cargo run --release --features lapack-openblas \
//!     --example five_me_cyclohexenone_1h
//! ```
//!
//! The example prints the active backend next to the eigendecomposition
//! timing, so diffing two runs immediately shows the speedup in the one
//! phase where it should appear. The FID loop is pure nalgebra complex
//! arithmetic in the eigenbasis and is identical across backends — a useful
//! sanity check that any timing improvement is surgical, not global.
//!
//! Writes `five_me_cyclohexenone_1h_spectrum.csv` in the working directory.

use nmr_sim::operator::total_m_minus;
use nmr_sim::{
    apodize_exponential, thermal_x_state, zero_fill, DiscreteSpectrum, EigenPropagator,
    Hamiltonian, Isotope, JCouplingH, Spin, SpinSystem, SumH, ZeemanH, EIGEN_BACKEND,
};

/// Field strength for this simulation.
const SPECTROMETER_MHZ: f64 = 600.0;
/// Conversion: 14.0954 T ↔ 600 MHz for 1H; scales linearly in field.
const B0_TESLA: f64 = 14.0954;

/// Chemical shifts (ppm) for groups A..H — from Lucas's assignment table.
/// Index into the returned spin list follows the A..G ring-proton ordering
/// with the three methyl protons occupying positions 7, 8, 9.
const SHIFTS_PPM: [(&str, f64); 8] = [
    ("A", 6.95),
    ("B", 6.01),
    ("C", 2.48),
    ("D", 2.42),
    ("E", 2.22),
    ("F", 2.12),
    ("G", 2.04),
    ("H", 1.07),
];

/// Methyl protons occupy three spin slots at the same shift. Intra-methyl
/// couplings are zero (equivalent protons do not split each other).
const METHYL_COUNT: usize = 3;

/// Group-level J-couplings (A..G only — methyl couplings handled separately
/// because they need to be replicated across all three methyl protons).
/// Format: `(group_i, group_j, J_Hz)` with `group_i < group_j` using the
/// A..G ordering above (A = 0, …, G = 6).
///
/// Twelve entries. Zero entries from the spreadsheet are omitted; the two
/// non-zero methyl (H-group) couplings are split out into `METHYL_COUPLINGS`
/// below so they can be replicated across all three methyl spins.
const GROUP_COUPLINGS: &[(usize, usize, f64)] = &[
    // A (index 0)
    (0, 1, 10.1), // A-B: endocyclic vinyl
    (0, 3, 5.7),  // A-D
    (0, 6, 2.7),  // A-G
    // B (index 1)
    (1, 3, 1.2), // B-D
    (1, 6, 2.7), // B-G
    // C (index 2)
    (2, 3, 1.3),  // C-D
    (2, 4, 3.7),  // C-E
    (2, 5, 16.1), // C-F: geminal CH₂
    // D (index 3)
    (3, 4, 5.4),  // D-E
    (3, 6, 18.6), // D-G: geminal CH₂
    // E (index 4)
    (4, 5, 12.4), // E-F
    (4, 6, 9.9),  // E-G
                  // (F-G, F-H, G-H all zero — omitted)
];

/// Methyl-to-ring couplings (the "H" group in the spreadsheet, index 7..=9
/// in the full spin list). Entries: `(ring_group, J_Hz_to_methyl)`.
const METHYL_COUPLINGS: &[(usize, f64)] = &[
    (2, 1.0), // C-H: long-range / allylic
    (4, 6.6), // E-H: the methine-to-methyl triplet splitting
];

/// Exponential line broadening (Hz) applied to the FID before FFT. 1 Hz is
/// the conventional default for 1D 1H: large enough that each peak spans
/// several FFT bins after zero-filling, small enough to leave J-splittings
/// resolved. When real T₂ relaxation lands this should drop to 0.
const LB_HZ: f64 = 1.0;
/// Zero-fill factor (Topspin default for 1D).
const ZF_FACTOR: usize = 4;

fn main() {
    // ---- Build the 10-spin system ----
    //
    // A..G → 1 proton each at their tabulated shift. H → 3 protons at 1.07 ppm.
    // Total: 7 + 3 = 10 spins, dim = 1024.
    let mut spins: Vec<Spin> = SHIFTS_PPM[..7]
        .iter()
        .map(|(label, ppm)| Spin::labeled(Isotope::H1, *ppm, *label))
        .collect();
    // Methyl: three equivalent protons at the same shift. The label
    // intentionally repeats so downstream tooling can identify the group.
    let methyl_shift = SHIFTS_PPM[7].1;
    for _ in 0..METHYL_COUNT {
        spins.push(Spin::labeled(Isotope::H1, methyl_shift, "H"));
    }
    let sys = SpinSystem::new(spins, B0_TESLA);
    println!(
        "System: {} spins, Hilbert dim = {}, B₀ = {:.4} T ({} MHz)",
        sys.len(),
        sys.dim(),
        B0_TESLA,
        SPECTROMETER_MHZ as u32,
    );

    // ---- Assemble the J-coupling list ----
    //
    // Group-level couplings go in as-is. Methyl couplings get replicated
    // once per methyl proton — each ring proton is coupled to every methyl
    // spin with the same J (this is equivalent to treating the methyl as a
    // single effective spin-3/2 group under first-order / equivalent-proton
    // averaging, but here we let the full quantum math play out).
    let mut couplings: Vec<(usize, usize, f64)> = GROUP_COUPLINGS.to_vec();
    for &(ring, j_hz) in METHYL_COUPLINGS {
        for m in 0..METHYL_COUNT {
            let methyl_idx = 7 + m;
            couplings.push((ring, methyl_idx, j_hz));
        }
    }
    println!("Total J-coupling pairs: {}", couplings.len());

    // ---- Hamiltonian: H_Z + H_J ----
    let h_zeeman = ZeemanH::new(&sys);
    let h_coupling = JCouplingH::new(&sys, &couplings);
    let h: Box<dyn Hamiltonian> =
        Box::new(SumH::new(vec![Box::new(h_zeeman), Box::new(h_coupling)]));
    println!("Hamiltonian dim: {}", h.dim());

    // ---- Propagator: EigenPropagator ----
    //
    // Δt chosen by Nyquist margin: 1H spectral content lives within ±~10 ppm
    // at 600 MHz = ±6 kHz; 20 kHz sampling leaves room.
    //
    // EigenPropagator does the work up front (one O(D³) diagonalization of a
    // real-symmetric 1024×1024 matrix) and then amortizes the cost across
    // every FID sample. The eigendecomposition is the only place where the
    // parallel-LAPACK feature changes anything — the backend label is
    // printed on its own line so the speedup is pinpointable.
    let dt: f64 = 1.0 / 20_000.0;
    println!("Eigendecomposition backend: {EIGEN_BACKEND}");
    println!("Building EigenPropagator (diagonalizing 1024×1024 real-symmetric H)…");
    let build_start = std::time::Instant::now();
    let propagator = EigenPropagator::new(h.as_ref(), dt);
    println!(
        "  …done in {:.3}s [{EIGEN_BACKEND}]",
        build_start.elapsed().as_secs_f64(),
    );

    // ---- FID ----
    //
    // Same sizing reasoning as the smaller examples: N = 65_536 × 50 μs =
    // 3.28 s of "acquisition", which at LB = 1 Hz leaves exp(−π · 3.28)
    // ≈ 3.5e-5 of the starting amplitude at the truncation — enough headroom
    // to eliminate boxcar-sinc ripples in absorption mode.
    //
    // Spectral resolution: 1/(N·Δt) = 0.305 Hz = 5×10⁻⁴ ppm at 600 MHz,
    // then ×4 zero-fill → 0.076 Hz = 1.3×10⁻⁴ ppm per bin. More than enough
    // to resolve every J-coupling in the system (smallest at 1.0 Hz).
    let n_points: usize = 65_536;
    let rho0 = thermal_x_state(&sys);
    let obs = total_m_minus(&sys);
    println!(
        "Computing FID ({} points, this is the long step)…",
        n_points
    );
    let fid_start = std::time::Instant::now();
    let mut fid = propagator.compute_fid(&rho0, &obs, n_points);
    println!(
        "  …done in {:.2}s ({:.0} samples/s)",
        fid_start.elapsed().as_secs_f64(),
        n_points as f64 / fid_start.elapsed().as_secs_f64(),
    );

    // ---- Processing: apodize then zero-fill ----
    apodize_exponential(&mut fid, dt, LB_HZ);
    let fid = zero_fill(fid, ZF_FACTOR);

    // ---- Spectrum ----
    let spectrum = DiscreteSpectrum::from_fid(&fid, dt);
    println!(
        "Spectrum: {} bins, Δf = {:.3} Hz ({:.5} ppm at {} MHz)",
        spectrum.len(),
        spectrum.freq_resolution_hz(),
        spectrum.freq_resolution_hz() / SPECTROMETER_MHZ,
        SPECTROMETER_MHZ as u32,
    );

    // Report the top 8 peaks — a decent "is anything sensible coming out"
    // glance. A proper qualitative comparison against the experimental
    // spectrum is a plotting task, not stdout.
    let ppms = spectrum.frequencies_ppm(Isotope::H1, B0_TESLA);
    let mags = spectrum.magnitude();
    let mut indexed: Vec<(usize, f64)> = mags.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top 8 peaks (by magnitude):");
    for (rank, (idx, mag)) in indexed.iter().take(8).enumerate() {
        println!(
            "  #{}: {:7.3} ppm   mag = {:.3e}",
            rank + 1,
            ppms[*idx],
            mag
        );
    }

    // ---- CSV output ----
    let filename = "five_me_cyclohexenone_1h_spectrum.csv";
    match spectrum.to_csv_ppm(filename, Isotope::H1, B0_TESLA) {
        Ok(()) => println!("Saved {filename}"),
        Err(e) => eprintln!("Error writing {filename}: {e}"),
    }

    println!(
        "\nPlot with:\n    python scripts/plot_spectrum.py {filename}\n\
         (default window −1 to 15 ppm — try `--xlim 0 8` to zoom the \
         aliphatic region)",
    );
}
