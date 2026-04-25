//! Field-sweep study: 5-methylcyclohex-2-en-1-one 1H spectrum from 40 MHz
//! to 1000 MHz in a geometric progression.
//!
//! # What this example does
//!
//! Runs the same 10-spin Hamiltonian as `five_me_cyclohexenone_1h` at every
//! spectrometer frequency in the sequence
//!
//! ```text
//! f_k = 40 MHz · FACTOR^k,   k = 0, 1, …, 100,   FACTOR = 25^(1/100) ≈ 1.03271
//! ```
//!
//! equivalently, `ln f_k` is uniformly spaced on `[ln 40, ln 1000]` in 101
//! steps. That gives 101 field points: 40, 41.31, 42.66, …, 1000 MHz. The
//! motivation is to visualize how the spectrum morphs as the Zeeman term
//! grows relative to the J-coupling term — i.e., the strong-coupling →
//! weak-coupling transition. At 40 MHz the largest J in this molecule
//! (18.6 Hz, geminal CH₂) is 7.8× Δν for the closest pair (C at 2.48 vs
//! D at 2.42 ppm, Δν = 2.4 Hz → J/Δν > 7, deeply strong), producing
//! dramatic roofing and 2nd-order multiplet reshuffling. At 1000 MHz the
//! same pair sits at Δν = 60 Hz → J/Δν = 0.31, firmly in the weak-coupling
//! regime where first-order Pascal's-triangle multiplets re-emerge. With
//! 101 frames the slider sweep is dense enough that the transition looks
//! continuous rather than stepwise.
//!
//! # Output
//!
//! Writes `examples/outputs/five_me_cyclohexenone_field_sweep/` containing:
//!
//! - `000_0040.00MHz.csv` … `100_1000.00MHz.csv` — one absorption-mode
//!   spectrum per field, all on a common ppm axis (see design note below).
//! - `manifest.csv` — (index, spectrometer_mhz, b0_tesla, filename) for the
//!   Python plotter to consume without having to parse filenames.
//!
//! Living under `examples/outputs/` (a single gitignored directory shared by
//! every example) keeps the repo root clean.
//!
//! # Design note — common ppm axis
//!
//! For an overlay plot to be meaningful, all 101 spectra must live on the
//! same ppm grid. The cleanest way to enforce that is to adapt `dt` per
//! run: choose `dt = 1 / (SW_ppm · spec_MHz)` so the FFT bin spacing in
//! Hz scales with the field and the bin spacing in ppm (which is Hz /
//! spec_MHz) is invariant. With SW_ppm = 33.333 and N = 32768, every run
//! produces a spectrum on ppm ∈ [−16.67, +16.67] with Δ = 0.001 ppm/bin
//! native (0.00025 ppm/bin after 4× zero-fill). Python can stack the 101
//! intensity arrays into a single (101, N_eff) matrix and share one x-axis
//! with zero interpolation.
//!
//! # Runtime
//!
//! Each field run is ~0.5 s eigendecomp + ~25 s FID at N = 32,768. Summed
//! CPU time across 101 fields is ~40–50 minutes, but the runs are fully
//! independent so we parallelise them with rayon — the wall clock drops to
//! roughly `summed_CPU / n_cores`. On an 8-core laptop expect ~6 min; on
//! a 16-core HPC node ~3 min. The example prints both wall and summed-CPU
//! time at the end, plus the realised speedup, so it's a self-checking
//! sanity gauge for the parallel setup.
//!
//! If you want a faster scouting pass, drop `N_POINTS` to 16384 — half the
//! native ppm resolution (still much finer than anything visible in the
//! plot), but cuts FID time in half.
//!
//! ## Parallelism gotcha — BLAS thread oversubscription
//!
//! If you also build with `--features lapack-accelerate` (or any
//! `lapack-*` backend), the eigendecomposition is *itself* multi-threaded.
//! Naively, rayon spawns N_cores tasks each spawning N_cores BLAS threads
//! → N_cores² way contention, and the wall clock can get worse rather
//! than better. The standard fix is "outer parallelism wins": pin the
//! BLAS thread pool to one before launching the example. Set whichever
//! env var matches the backend you compiled in:
//!
//! ```sh
//! # Linux + OpenBLAS
//! OPENBLAS_NUM_THREADS=1 cargo run --release \
//!     --features lapack-openblas \
//!     --example five_me_cyclohexenone_field_sweep
//!
//! # macOS + Accelerate
//! VECLIB_MAXIMUM_THREADS=1 cargo run --release \
//!     --features lapack-accelerate \
//!     --example five_me_cyclohexenone_field_sweep
//!
//! # Linux + MKL
//! MKL_NUM_THREADS=1 cargo run --release \
//!     --features lapack-intel-mkl \
//!     --example five_me_cyclohexenone_field_sweep
//! ```
//!
//! With no LAPACK feature (the pure-Rust default), nalgebra's
//! `symmetric_eigen` is single-threaded already and there's no
//! oversubscription concern.
//!
//! Run with:
//!
//! ```sh
//! # Pure Rust (default) — rayon owns all the parallelism, no env vars needed.
//! cargo run --release --example five_me_cyclohexenone_field_sweep
//!
//! # Then plot:
//! python scripts/plot_field_sweep.py examples/outputs/five_me_cyclohexenone_field_sweep/
//! ```

use std::fs::{create_dir_all, File};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use nmr_sim::operator::total_m_minus;
use nmr_sim::{
    apodize_exponential, thermal_x_state, zero_fill, DiscreteSpectrum, EigenPropagator,
    Hamiltonian, Isotope, JCouplingH, Spin, SpinSystem, SumH, ZeemanH,
};

// ---------- Sweep parameters ----------

/// Geometric common ratio between successive field points. Chosen so that
/// MIN_MHZ × FACTOR^100 = MAX_MHZ exactly. Equivalent definitions:
///   FACTOR = 25^(1/100) = exp(ln(MAX_MHZ / MIN_MHZ) / 100) = exp(0.0321887…)
const FACTOR: f64 = 1.032_712_419_9;
const MIN_MHZ: f64 = 40.0;
const MAX_MHZ: f64 = 1000.0;
/// Number of points in the sweep — `MIN_MHZ · FACTOR^(N_FIELDS-1) = MAX_MHZ`,
/// so 101 points means 100 steps of size `a = ln(FACTOR)` in log-frequency.
const N_FIELDS: usize = 101;

/// Spectral window in ppm — kept constant across all fields so the ppm bin
/// spacing is identical for every run. 33.333 ppm matches the 20 kHz SW the
/// single-field example uses at 600 MHz.
const SW_PPM: f64 = 33.333_333_333_333_33;

/// FID length (time-domain samples before zero-fill). 32,768 is half the
/// single-field example's N; the resulting native resolution (0.001 ppm/bin)
/// is still finer than anything visible in the plot. At 101 fields this is
/// ~40–50 min total — drop to 16,384 for a ~25 min scouting pass.
const N_POINTS: usize = 32_768;

/// Exponential line broadening applied to each FID before FFT. At LB = 1 Hz
/// the native 1 Hz long-range C-H coupling is still resolved.
const LB_HZ: f64 = 1.0;

/// Zero-fill factor — same as the single-field example.
const ZF_FACTOR: usize = 4;

/// Output directory (created if absent). Sits under the repo-wide
/// `examples/outputs/` umbrella so a single gitignore line covers every
/// example's generated files.
const OUTPUT_DIR: &str = "examples/outputs/five_me_cyclohexenone_field_sweep";

// ---------- Spin system (mirrors five_me_cyclohexenone_1h.rs) ----------

/// Chemical shifts (ppm) for groups A..H — from Lucas's assignment table.
/// Deliberately duplicated from the single-field example rather than lifted
/// into a shared module: examples in Rust don't share code gracefully, and
/// colocating the data with the simulation it drives makes each example
/// readable top-to-bottom.
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

const METHYL_COUNT: usize = 3;

/// Group-level J-couplings (A..G only). See the single-field example for
/// commentary on each entry.
const GROUP_COUPLINGS: &[(usize, usize, f64)] = &[
    (0, 1, 10.1),
    (0, 3, 5.7),
    (0, 6, 2.7),
    (1, 3, 1.2),
    (1, 6, 2.7),
    (2, 3, 1.3),
    (2, 4, 3.7),
    (2, 5, 16.1),
    (3, 4, 5.4),
    (3, 6, 18.6),
    (4, 5, 12.4),
    (4, 6, 9.9),
];

/// Methyl-to-ring couplings, replicated across all three methyl spins.
const METHYL_COUPLINGS: &[(usize, f64)] = &[(2, 1.0), (4, 6.6)];

// ---------- Helpers ----------

/// Convert 1H spectrometer frequency (MHz) to field strength (T) via the
/// relation 2πν = γ_H · B₀. `Isotope::H1::gamma()` is `const fn`, so this
/// whole computation is const-evaluable at call sites (we just don't need
/// it to be in a loop).
fn mhz_to_tesla(spec_mhz: f64) -> f64 {
    2.0 * std::f64::consts::PI * spec_mhz * 1e6 / Isotope::H1.gamma()
}

/// Build the 101-point geometric sweep. We compute each point as
/// `exp(ln(MIN_MHZ) + k · a)` rather than chaining `f *= FACTOR`, because
/// the additive form in log-space accumulates only one rounding per point
/// instead of N — the endpoints land on MIN and MAX to f64 precision
/// without any post-hoc snapping.
fn build_frequency_sweep() -> Vec<f64> {
    let ln_min = MIN_MHZ.ln();
    let ln_max = MAX_MHZ.ln();
    let a = (ln_max - ln_min) / ((N_FIELDS - 1) as f64);
    (0..N_FIELDS)
        .map(|k| (ln_min + (k as f64) * a).exp())
        .collect()
}

/// Build the 10-spin system for a given field. Uses the same A..G + 3 methyl
/// layout as the single-field example.
fn build_spin_system(b0_tesla: f64) -> SpinSystem {
    let mut spins: Vec<Spin> = SHIFTS_PPM[..7]
        .iter()
        .map(|(label, ppm)| Spin::labeled(Isotope::H1, *ppm, *label))
        .collect();
    let methyl_shift = SHIFTS_PPM[7].1;
    for _ in 0..METHYL_COUNT {
        spins.push(Spin::labeled(Isotope::H1, methyl_shift, "H"));
    }
    SpinSystem::new(spins, b0_tesla)
}

/// Expand the group-level + methyl-replicated J-coupling list into the
/// flat (i, j, J_Hz) form that `JCouplingH::new` consumes.
fn build_coupling_list() -> Vec<(usize, usize, f64)> {
    let mut couplings: Vec<(usize, usize, f64)> = GROUP_COUPLINGS.to_vec();
    for &(ring, j_hz) in METHYL_COUPLINGS {
        for m in 0..METHYL_COUNT {
            couplings.push((ring, 7 + m, j_hz));
        }
    }
    couplings
}

// ---------- Per-field simulation ----------

/// Simulate one spectrum at the given field and write it to `path`.
/// Returns (wall_time_eigen_s, wall_time_fid_s) for logging.
fn simulate_and_save(spec_mhz: f64, path: &str) -> (f64, f64) {
    let b0 = mhz_to_tesla(spec_mhz);
    // Adaptive dt: keeps the ppm bin spacing (and therefore the full ppm
    // window SW_PPM) identical across all fields. Derivation:
    //   SW_Hz = 1/dt,  SW_ppm = 1e6 · SW_Hz / spec_Hz
    //         = 1e6 / (dt · spec_mhz · 1e6) = 1 / (dt · spec_mhz)
    // → dt = 1 / (SW_PPM · spec_mhz)
    // Sanity check: at 600 MHz this gives dt = 50 μs, matching the single-
    // field example's 20 kHz SW.
    let dt = 1.0 / (SW_PPM * spec_mhz);

    let sys = build_spin_system(b0);
    let couplings = build_coupling_list();
    let h: Box<dyn Hamiltonian> = Box::new(SumH::new(vec![
        Box::new(ZeemanH::new(&sys)),
        Box::new(JCouplingH::new(&sys, &couplings)),
    ]));

    let eigen_start = std::time::Instant::now();
    let propagator = EigenPropagator::new(h.as_ref(), dt);
    let eigen_s = eigen_start.elapsed().as_secs_f64();

    let rho0 = thermal_x_state(&sys);
    let obs = total_m_minus(&sys);

    let fid_start = std::time::Instant::now();
    let mut fid = propagator.compute_fid(&rho0, &obs, N_POINTS);
    let fid_s = fid_start.elapsed().as_secs_f64();

    apodize_exponential(&mut fid, dt, LB_HZ);
    let fid = zero_fill(fid, ZF_FACTOR);

    let spectrum = DiscreteSpectrum::from_fid(&fid, dt);
    spectrum
        .to_csv_ppm(path, Isotope::H1, b0)
        .expect("failed to write spectrum CSV");

    (eigen_s, fid_s)
}

// ---------- Driver ----------

/// One row of completed-simulation metadata, collected back from the rayon
/// pool and used to write `manifest.csv` in canonical order at the end.
struct FieldResult {
    index: usize,
    spec_mhz: f64,
    b0_tesla: f64,
    filename: String,
    eigen_s: f64,
    fid_s: f64,
}

fn main() {
    let freqs = build_frequency_sweep();
    let total = freqs.len();
    println!(
        "Field sweep: {} frequencies from {:.2} MHz to {:.2} MHz (factor = {:.6})",
        total,
        freqs[0],
        freqs[total - 1],
        FACTOR,
    );
    println!(
        "N = {}, ZF = {}×, LB = {:.1} Hz, SW = {:.2} ppm (common across all fields)",
        N_POINTS, ZF_FACTOR, LB_HZ, SW_PPM,
    );
    println!(
        "Parallelism: rayon ({} threads). Tasks complete out of order; the \
         counter below reports monotonic finish progress, not field index.",
        rayon::current_num_threads(),
    );
    println!();

    create_dir_all(OUTPUT_DIR).expect("failed to create output directory");

    // Atomic counter for the "k of N done" progress display. Threads call
    // `fetch_add` as they finish, so the printed counter is always monotonic
    // even though the field indices arrive in whatever order the rayon pool
    // happens to schedule them.
    let done = AtomicUsize::new(0);

    let sweep_start = std::time::Instant::now();

    // Run every field's simulation in parallel. Each task is fully self-
    // contained — builds its own Hamiltonian, eigendecomposes, propagates,
    // writes its CSV — so there's nothing to share except the per-task
    // input frequency. The collected `Vec<FieldResult>` is the only join
    // point; the manifest is written serially below.
    let mut results: Vec<FieldResult> = freqs
        .par_iter()
        .enumerate()
        .map(|(i, &spec_mhz)| {
            let b0 = mhz_to_tesla(spec_mhz);
            // Zero-padded index (3 digits, 0..100) + 7-wide MHz with leading
            // zeros (e.g. "067_0317.97MHz") makes the directory listing sort
            // correctly by field. 7 = 4 integer digits + dot + 2 decimals,
            // big enough for 1000.00.
            let filename = format!("{:03}_{:07.2}MHz.csv", i, spec_mhz);
            let path = format!("{}/{}", OUTPUT_DIR, filename);

            let (eigen_s, fid_s) = simulate_and_save(spec_mhz, &path);

            // Bump the counter after the work, so "k done" reflects actual
            // completions, not task starts.
            let k = done.fetch_add(1, Ordering::Relaxed) + 1;
            // Single println! is atomic w.r.t. line interleaving — multiple
            // threads can call it concurrently and you'll get whole lines,
            // never half-lines spliced together.
            println!(
                "[{:>3}/{}] field #{:03} {:7.2} MHz (B₀ = {:.4} T) → eigen {:.2}s, FID {:.1}s, {}",
                k, total, i, spec_mhz, b0, eigen_s, fid_s, filename,
            );

            FieldResult {
                index: i,
                spec_mhz,
                b0_tesla: b0,
                filename,
                eigen_s,
                fid_s,
            }
        })
        .collect();

    // Sort by field index so the manifest is canonical regardless of finish
    // order. The Python plotter sorts independently, so this is belt-and-
    // braces, but it makes the file inspectable by hand.
    results.sort_by_key(|r| r.index);

    // Manifest: one row per field. Written *after* the parallel section so
    // we never need to lock a shared file handle.
    let manifest_path = format!("{}/manifest.csv", OUTPUT_DIR);
    let mut manifest = File::create(&manifest_path).expect("failed to create manifest");
    writeln!(manifest, "index,spectrometer_mhz,b0_tesla,filename").unwrap();
    for r in &results {
        writeln!(
            manifest,
            "{},{:.6},{:.6},{}",
            r.index, r.spec_mhz, r.b0_tesla, r.filename,
        )
        .unwrap();
    }

    // Aggregate timing — interesting because the wall clock should be much
    // less than the summed CPU time once rayon is doing its job.
    let wall_s = sweep_start.elapsed().as_secs_f64();
    let cpu_eigen_s: f64 = results.iter().map(|r| r.eigen_s).sum();
    let cpu_fid_s: f64 = results.iter().map(|r| r.fid_s).sum();
    let cpu_total_s = cpu_eigen_s + cpu_fid_s;

    println!();
    println!(
        "Done in {:.1}s wall (vs {:.1}s summed CPU; speedup ≈ {:.1}×). Output: {}/",
        wall_s,
        cpu_total_s,
        cpu_total_s / wall_s,
        OUTPUT_DIR,
    );
    println!();
    println!("Plot the overlay with the slider:");
    println!("    python scripts/plot_field_sweep.py {}/", OUTPUT_DIR);
}
