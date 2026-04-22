# nmr-sim — Architecture & Design

_Last updated: 2026-04-22. This document is the single source of truth for
the shape of the project. If the code disagrees with the doc, we either fix
the code or update the doc — never let them drift silently._

---

## 1. Vision

`nmr-sim` is intended to become the best-in-class open-source platform for
simulating, analyzing, and learning from NMR data. Its ultimate target is a
Bayesian machine-learning pipeline that predicts small-molecule structure
from NMR spectra. That long-term goal requires, in order:

1. A **fast, correct spin dynamics simulator** (density matrix / Liouville
   space) for arbitrary spin systems.
2. **I/O for real spectrometer data** — Bruker TopSpin and Varian/Agilent
   VNMRJ formats, plus the portable formats (JCAMP-DX, nmrML).
3. A **database** that stores experimental spectra and simulation parameters
   alongside simulated spectra and their derived features.
4. A **GUI** that lets scientists upload raw data, visualize experimental
   vs. predicted spectra, and drive simulations without writing Rust.
5. A **Bayesian inference loop** that treats each experiment as evidence and
   updates an ML model of spectrum → structure.

We are under no illusion that this is a small project. We are nevertheless
building toward it.

## 2. Design principles

- **Physics first, ergonomics close behind.** The simulator must be correct
  for problems it claims to solve. Users should not have to tune numerical
  parameters to get standard spectra right.
- **Rust idioms, not a MATLAB port.** Strong types, traits for extensibility,
  zero-cost abstractions where they help, `Result<T, E>` for anything that
  can fail. No `unsafe`.
- **Separation of concerns, enforced by the crate layout.** The simulation
  engine should not know about files, the GUI should not know about
  Hamiltonians, and the ML pipeline should not know about byte layouts of
  Bruker's `acqus`.
- **Benchmark-driven performance.** "Fast" is a testable claim. Every major
  kernel gets a Criterion bench.
- **Validate against the state of the art.** Spinach, SIMPSON, SpinDynamica,
  NMRPipe, and published reference tables are our oracles.
- **Public from day one.** Dual MIT / Apache-2.0 license, CI gating every
  merge, architecture written down where contributors can find it.
- **Preserve what works.** The existing first-order simulator is not a
  throwaway — it becomes the `backend::first_order` path, usable as a fast
  approximation and as a test oracle for the quantum engine on weakly-coupled
  problems.

## 3. Current state (v0.1.0)

One crate: `nmr-sim`. Public API:

| Type                         | Role                                                         |
|------------------------------|--------------------------------------------------------------|
| `Peak`                       | One Lorentzian line (shift, area, linewidth).                |
| `Coupling`                   | A scalar J-coupling to `n_neighbors` equivalent nuclei.      |
| `PeakDef`                    | A site before J-coupling expansion.                          |
| `expand_peak`, `pascal_row`  | First-order multiplet expansion math.                        |
| `LorentzianTemplate`         | Pre-computed cos²(θ) template for fast Lorentzian eval.      |
| `Spectrum`                   | A collection of peaks on a frequency grid; `to_csv(...)`.    |

Physics model: **first-order multiplets only.** No strong coupling, no
relaxation evolution, no pulse sequences, no 2D, no heteronuclear mixing,
no solid-state. Sufficient for most routine 1H small-molecule spectra at
high field; insufficient for the rest of the project's goals.

Notable engineering detail: the Lorentzian is evaluated via the substitution
`x = tan(θ)`, which converts the peaked `1 / (x² + 1)` into the smooth
`cos²(θ)`. Linear interpolation in θ-space is therefore accurate even in
the peak's sharp core. See `docs/lorentzian_template_guide.html`.

## 4. Crate roadmap

The workspace will evolve as follows. Crates become real when there's enough
code to justify the split — we don't create empty crates.

| Crate           | Role                                                | Status       |
|-----------------|-----------------------------------------------------|--------------|
| `nmr-sim`       | Simulation engines (first-order + quantum).         | **Live.**    |
| `nmr-core`      | Shared types: spin systems, operators, Hamiltonians.| Planned — may stay inside `nmr-sim` until there's a second consumer. |
| `nmr-io`        | Bruker, Varian, JCAMP-DX, nmrML readers/writers.    | Planned (Phase 3). |
| `nmr-analysis` | Peak picking, integration, deconvolution, fitting.  | Planned.     |
| `nmr-cli`       | Thin command-line wrapper; renders spectra to file. | Planned once there's more than the example binary. |
| `nmr-db`        | Database abstraction for experimental + simulated.  | Planned (Phase 4). |
| `nmr-gui`       | Desktop GUI (likely Tauri + egui/dioxus, TBD).      | Planned (Phase 4). |
| `nmr-ml`        | Feature extraction, Bayesian loop, NN interface.    | Planned (Phase 5). |

**Heuristic:** don't split a crate out until (a) it has at least one
non-test consumer, or (b) it has independent CI needs (e.g., GPU-specific
dependencies, native linker flags). Otherwise module boundaries are cheaper
than crate boundaries and give the same encapsulation benefits.

## 5. Core abstractions (Phase 1 sketch)

This is the proposed trait hierarchy for the quantum engine. **Nothing in
this section is coded yet; all of it is up for debate.**

### 5.1 SpinSystem

```rust
/// A collection of spins with their interactions.
pub struct SpinSystem {
    spins: Vec<Spin>,                 // index = spin label
    couplings: Vec<ScalarCoupling>,   // J in Hz between pairs
    dipolar: Vec<DipolarCoupling>,    // b_IS in Hz, with orientation
    csa: Vec<ChemicalShiftAnisotropy>,
}

pub struct Spin {
    pub isotope: Isotope,    // 1H, 13C, 15N, 19F, 31P, 2H, …
    pub iso_shift: f64,      // isotropic chemical shift, in ppm
    pub label: String,       // human-readable identifier
}
```

- `Isotope` carries nuclear spin (I = 1/2, 1, 3/2, …), gyromagnetic ratio,
  natural abundance. Looked up from a table at construction time; stored by
  value (it's a few `f64`s).
- Interactions are stored sparsely, addressed by spin index, not by name.
- `SpinSystem` is *plain data* — no propagation logic lives here.

### 5.2 Operators

We need Hermitian spin operators (Sx, Sy, Sz, I⁺, I⁻) for each spin, and
efficient products and sums of them.

For Phase 1 (small systems: up to ~10 spin-1/2 ≈ 1024-dimensional Hilbert
space), dense complex matrices via `nalgebra` or `ndarray` are fine.

```rust
pub type Complex = num_complex::Complex<f64>;
pub type DMatrix = nalgebra::DMatrix<Complex>;

pub trait Operator {
    fn matrix(&self, system: &SpinSystem, basis: &Basis) -> DMatrix;
}

pub enum SpinAxis { X, Y, Z, Plus, Minus }

pub struct SingleSpinOp { pub spin: usize, pub axis: SpinAxis }
pub struct OperatorSum(pub Vec<(Complex, Box<dyn Operator>)>);
pub struct OperatorProduct(pub Vec<Box<dyn Operator>>);
```

The `Basis` parameter exists because later we may want Zeeman (computational)
vs. eigen-basis vs. Liouville basis; Phase 1 will support one and leave the
hook for more.

### 5.3 Hamiltonian

```rust
pub trait Hamiltonian {
    fn matrix(&self, system: &SpinSystem, basis: &Basis) -> DMatrix;
}
```

Concrete Hamiltonians:

- `Zeeman(b0)` — isotropic chemical-shift Hamiltonian in the rotating frame.
- `ScalarCouplingH` — Σ 2π J_ij I_i · I_j (weak: I_iz I_jz; strong: full dot).
- `DipolarH`, `CSAH`, `QuadrupolarH` — later.
- `PulseRF(phase, field_strength)` — the time-dependent driving term during
  a pulse.

A full experiment Hamiltonian is an `OperatorSum` of these pieces.

### 5.4 Propagator

```rust
pub trait Propagator {
    /// Evolve a density matrix by time `dt` under some dynamics.
    fn step(&self, rho: &mut DMatrix, dt: f64);
}
```

Phase-1 implementations:

- `MatrixExpPropagator` — straightforward `exp(-i H dt / ħ)` via
  diagonalization (cheap in Hilbert space for H up to ~1024×1024).
- `LiouvillePropagator` — same story in Liouville space, needed for open-
  system dynamics (relaxation, exchange).
- `KrylovPropagator` — Lanczos / Arnoldi-based `exp(-i H dt)` for larger
  systems where eigendecomp is the bottleneck. Deferred until profiling
  says we need it.

### 5.5 Experiments

An experiment is a sequence of operations (delays, pulses, detection) that
produces a signal.

```rust
pub trait Experiment {
    fn run(&self, system: &SpinSystem) -> Signal;
}

pub struct PulseAcquire { /* 90° hard pulse, then acquire */ }
pub struct Cosy { /* 90° – t1 – 90° – t2 */ }
pub struct Hsqc { /* heteronuclear 1H–13C single-quantum */ }
```

The library should make it easy to *compose* pulse sequences from primitives
(pulse, delay, gradient, phase cycle) rather than ship only a fixed catalog.

### 5.6 Signal & Spectrum

```rust
pub struct Signal {
    time_hz: Vec<f64>,          // sample times, in seconds
    fid: Vec<Complex>,          // complex time-domain signal
    spectrometer_freq: f64,     // MHz
    isotope: Isotope,           // for ppm conversion
}

pub struct SpectrumFreq { /* Fourier-transformed, phase- and baseline-corrected */ }

impl Signal {
    pub fn fft(&self, opts: FftOptions) -> SpectrumFreq;
    pub fn apodize(&mut self, window: Window);
}
```

The existing `Spectrum` type will likely be renamed or wrapped to clarify
that it's the *frequency-domain* object; time-domain signals need their own
home.

## 6. Physics scope — Phase 1 targets

| Capability                                   | Phase 1 | Phase 2 | Later  |
|----------------------------------------------|:-------:|:-------:|:------:|
| Isotropic chemical shifts                    | ✅      |         |        |
| Scalar J-coupling (weak & strong)            | ✅      |         |        |
| Arbitrary spin-1/2 systems, dense Hilbert    | ✅      |         |        |
| Hard pulses (ideal 90°, 180°, arbitrary θ)   | ✅      |         |        |
| 1D pulse-acquire                             | ✅      |         |        |
| Basic 2D (COSY)                              | ✅      |         |        |
| Heteronuclear (HSQC, HMBC)                   |         | ✅      |        |
| Redfield / Lindblad relaxation               |         | ✅      |        |
| Chemical exchange (Bloch-McConnell, Liouville) |       | ✅      |        |
| Arbitrary spin (I > 1/2)                     |         | ✅      |        |
| Shaped pulses                                |         | ✅      |        |
| Gradients, coherence selection               |         | ✅      |        |
| MAS solid-state averaging                    |         |         | ✅     |
| Quadrupolar second-order effects             |         |         | ✅     |
| GPU acceleration                             |         |         | ✅     |

**Critical invariant for Phase 1:** whatever the first-order code computes
should match the quantum engine within numerical tolerance on problems
where the first-order model is exact (truly weak couplings, no mixing). This
is an automated cross-check.

## 7. Numerical strategy

- **Dense linear algebra:** `nalgebra` with its built-in LAPACK binding (or
  `ndarray` + `ndarray-linalg`; to decide). Complex support via
  `num-complex`. Decision gate: whichever has cleaner eigendecomposition
  ergonomics for complex Hermitian matrices at our sizes.
- **Sparse?** Probably not in Phase 1. Spin operators have known sparsity
  but dense matmul will be plenty fast up to ~12 spin-1/2 (4096² complex
  matrices — ~270 MiB, on the edge; 10-spin is 1024² = 16 MiB, comfortable).
- **Matrix exponentials:** start with explicit diagonalization
  (`exp(H) = V exp(D) V⁻¹`). Switch to Krylov only when sizes grow.
- **Parallelism:** `rayon` for obvious embarrassingly parallel sweeps
  (powder averaging, Monte Carlo over chemical-shift ensembles, parameter
  fits).
- **Precision:** `f64` everywhere. No configurable `f32` for simulation
  — output analysis / visualization can downcast.

## 8. I/O plan (Phase 3 preview)

| Format     | Role                                           | Parser crate?                |
|------------|------------------------------------------------|------------------------------|
| Bruker     | `acqus`, `procs`, `fid`, `1r`, `1i`, `pdata/`  | Custom; no maintained Rust crate exists. |
| Varian/Agilent | `procpar`, `fid`                           | Custom.                      |
| JCAMP-DX   | Portable ASCII spectra.                        | Custom.                      |
| nmrML      | XML, mzML-style.                               | `quick-xml` + custom types.  |
| CSV        | Trivial; already have.                         | `csv` crate.                 |

The I/O layer exposes `Signal` (time-domain) and `SpectrumFreq`
(processed). It should be careful never to throw away metadata — magnetic
field, pulse sequence parameters, acquisition parameters, and sample
description all flow through unchanged.

## 9. Database sketch (Phase 4 preview)

Stores three kinds of objects, linked by foreign keys:

1. **Experiments:** raw + processed data with all spectrometer metadata.
2. **Simulation runs:** input spin system, Hamiltonian terms, pulse
   sequence, output spectrum, the `nmr-sim` git commit that produced it
   (for reproducibility).
3. **Reference comparisons:** (experiment_id, simulation_id, metric_score).

Likely PostgreSQL (with `pgvector` for spectral similarity) via `sqlx`.
Questions to answer later: do we store spectra as BLOB, as Arrow Parquet,
or columnar in a separate time-series store?

## 10. GUI architecture (Phase 4 preview)

Working assumption: a Tauri desktop app with a Rust backend that links the
`nmr-*` crates directly (no FFI boundary for the core work) and a web UI.
UI framework TBD (`dioxus`, `leptos`, or plain web with React/Svelte).
Alternative: an `egui` single-binary app for simplicity; downside is weaker
typography and polish for publication-quality plots.

The GUI's job description:

1. Ingest raw Bruker/Varian folders (drag-and-drop).
2. Interactive spectrum viewer — zoom, integration, peak picking.
3. Build a spin-system hypothesis (structure or parameter-based).
4. Run simulations, overlay on experimental data.
5. Feed selected (experiment, simulation, label) triples into the ML loop.

## 11. ML pipeline (Phase 5 preview — sketch)

Deliberately vague. Candidate shape:

- Feature extractor: spectrum → fixed-size representation (CNN on the
  frequency axis? Transformer on peak lists? Both?).
- Bayesian loop: Given a candidate molecular structure, simulate its
  spectrum, compare to experimental, compute a likelihood. Use the corpus
  of (simulated, experimental, structure) triples to train / fine-tune a
  generative model over structures given spectra.
- Interface: ML code lives in Python (via PyO3 bindings back to
  `nmr-sim`) or in Rust (via `candle` / `burn`) — to be decided when we're
  close.

We will *not* pretend to have this designed today. The job in Phases 1–4 is
to generate the infrastructure and data that makes this phase tractable.

## 12. Testing & validation strategy

Three layers:

1. **Unit tests.** Standard Rust `#[test]` in each module. Every pure
   function gets a test with an analytically-known answer.
2. **Property tests.** `proptest` for invariants: area conservation under
   J-coupling expansion, trace preservation under unitary propagation,
   hermiticity of Hamiltonians.
3. **Cross-simulator validation.** Reference spectra computed in Spinach
   and/or SIMPSON, checked into `crates/nmr-sim/tests/fixtures/`. These
   run in CI as slower integration tests behind a feature flag
   (`--features validation`) so they don't bloat every build.

Spectral comparison uses a metric robust to small shift offsets — e.g.,
cosine similarity on low-pass filtered spectra, or earth-mover's distance
between peak lists. We pick one and commit in Phase 1.

## 13. Near-term milestones

Concrete, prioritized, revisable:

**M0 (this session, 2026-04-22):**
- [x] Cargo workspace layout.
- [x] Dual MIT/Apache-2.0 license.
- [x] README, CONTRIBUTING, .gitignore.
- [x] GitHub Actions CI (fmt, clippy, test, docs, MSRV build).
- [x] This architecture doc.
- [x] Local git repo initialized.

**M1 — Library shape (target: +1 week):**
- [ ] First pass at `nmr_sim::SpinSystem`, `Isotope`, `Spin` data types.
- [ ] `Operator` and `Hamiltonian` traits + first concrete implementations
      (single-spin Sx/Sy/Sz via `DMatrix<Complex>`, `ZeemanH`, `ScalarCouplingH`).
- [ ] `cargo doc` tour of the new surface.
- [ ] `nalgebra` + `num-complex` added as deps; corresponding version
      pins; spin operators validated against hand-computed matrices.

**M2 — First quantum spectrum (target: +2–3 weeks):**
- [ ] `MatrixExpPropagator` via eigendecomp.
- [ ] `PulseAcquire` experiment.
- [ ] Detect `⟨F⁺⟩(t)` as the FID; FFT to spectrum.
- [ ] Reproduce the first-order ibuprofen spectrum using the quantum
      engine (validates the stack end-to-end on a benign example).

**M3 — Strong coupling (target: +4 weeks):**
- [ ] Simulate a classic AB / ABX system; verify roofing effect and
      second-order coupling constants against Bruker's `Topspin` or
      published references.
- [ ] Cross-check first-order vs. quantum: equal in the weak-coupling
      limit, systematic deviation as couplings approach shift differences.

**M4 — 2D COSY (target: +6–8 weeks):**
- [ ] `Cosy` experiment.
- [ ] First correlation spectrum; compare against a literature reference.

After M4 we re-scope — the next question will be whether to go deeper
into physics (heteronuclei, relaxation) or start I/O work so we can feed
real experimental data back into the stack.

---

_This document should grow with the code. Every substantive PR that
changes abstractions should update the relevant section here as part
of the PR — see `CONTRIBUTING.md`._
