# A guided tour of `examples/ab_system_1h.rs` (M3: J-coupling)

**Audience:** you, six months from now, coming back to understand what M3 added on top of the Zeeman-only quantum engine.

This is the companion to [`walkthrough_zeeman_1h.md`](walkthrough_zeeman_1h.md). That one explained the full pipeline from `SpinSystem` to a ppm axis for a non-interacting system (three singlets). This one covers the M3 additions — `JCouplingH`, `SumH`, and the automatic routing through `MatrixPropagator` — by walking through the `ab_system_1h` example line by line.

We won't re-explain anything that's already covered in the Zeeman walkthrough. If a word or a module doesn't ring a bell, read that doc first.

---

## 0. What the AB system demonstrates

A two-proton AB system is the simplest NMR experiment where the quantum engine pulls ahead of a first-order simulator. The spectrum depends on the dimensionless ratio

$$
\rho = \frac{J}{|\nu_A - \nu_B|}
$$

- When $\rho \ll 1$ (weak coupling), the spectrum is two equal-intensity doublets, the same thing a first-order engine would draw.
- When $\rho \sim 1$ (strong coupling), you see a four-line AB quartet with pronounced "roofing" (inner lines tall, outer lines short) and peak positions that *don't* sit at $\nu_A \pm J/2$, $\nu_B \pm J/2$ anymore.
- When $\rho \to \infty$, the four lines collapse into a singlet at the chemical-shift average.

The example holds chemical shifts (in ppm) and $J$ (in Hz) fixed and scans $B_0$ — which is how a real spectroscopist walks this axis in the lab, because higher-field magnets push $\rho$ toward zero.

---

## 1. What's new in M3

Three code additions, all in `hamiltonian.rs`:

### 1.1 `JCouplingH`

```rust
pub struct JCouplingH { /* cached dense matrix */ }

impl JCouplingH {
    pub fn new(sys: &SpinSystem, couplings: &[(usize, usize, f64)]) -> Self { /* … */ }
}
```

Takes a list of `(spin_i, spin_j, J_hz)` triples and builds

$$
\hat H_J = \sum_{i<j} 2\pi J_{ij}\,\hat I_i \cdot \hat I_j
       = \sum_{i<j} 2\pi J_{ij}\,(\hat I_{x,i}\hat I_{x,j} + \hat I_{y,i}\hat I_{y,j} + \hat I_{z,i}\hat I_{z,j})
$$

by calling `operator::ix_at`, `iy_at`, `iz_at` (the lifted single-site operators from the Zeeman walkthrough) and summing the three-term dot product, scaled by $2\pi J$ to get rad/s.

**The key trait choice**: `JCouplingH` implements `Hamiltonian::as_dense` but leaves `try_as_diagonal` as the trait default, which returns `None`. Concretely:

```rust
impl Hamiltonian for JCouplingH {
    fn dim(&self) -> usize { self.dense.nrows() }
    fn as_dense(&self) -> Operator { self.dense.clone() }
    // try_as_diagonal: default None.
}
```

That `None` is load-bearing. It's the one-word statement that this Hamiltonian carries off-diagonal flip-flop terms ($\hat I_x\hat I_x + \hat I_y\hat I_y$ = $\tfrac{1}{2}(\hat I^+\hat I^- + \hat I^-\hat I^+)$) — which means any propagator that assumes diagonal storage won't work. The trait machinery already knows how to react to this: `DiagonalPropagator::new` calls `try_as_diagonal`, gets `None`, and panics with a helpful message; `MatrixPropagator::new` calls `as_dense` and always works. The type system pushes callers toward the right choice.

### 1.2 `SumH`

```rust
pub struct SumH { /* Vec<Box<dyn Hamiltonian>> + cached diagonal if possible */ }

impl SumH {
    pub fn new(terms: Vec<Box<dyn Hamiltonian>>) -> Self { /* … */ }
}
```

The additive combinator. This is where the trait design starts paying rent. You write your total Hamiltonian as

```rust
let h = SumH::new(vec![
    Box::new(ZeemanH::new(&sys)),
    Box::new(JCouplingH::new(&sys, &[(0, 1, j_hz)])),
]);
```

and the combinator takes care of the bookkeeping:

- `as_dense` returns the elementwise sum of each child's `as_dense`.
- `try_as_diagonal` returns `Some(cached_diagonal)` if and only if every child advertised a diagonal. That means a sum of two `ZeemanH`s still qualifies for the $O(D)$-storage fast path; a sum that contains one `JCouplingH` does not.

The nice invariant: adding more physics — relaxation, exchange, dipolar, CSA — doesn't require touching `SumH` or `ZeemanH` or `JCouplingH`. You implement the new term as a `Hamiltonian`, `Box` it, and push it into the `Vec`. If the new term's `try_as_diagonal` returns `Some`, `SumH` picks up the fast path automatically; otherwise everyone routes through `MatrixPropagator`.

### 1.3 No changes above `Hamiltonian`

`compute_fid`, `DiscreteSpectrum`, and the CSV output all still work unchanged. The only thing the example had to swap out was `DiagonalPropagator` → `MatrixPropagator`, and even that could have been avoided with a trait-object dispatch if we'd cared to write it. The whole pipeline is *backend-agnostic* below `Hamiltonian`.

---

## 2. Line-by-line through `examples/ab_system_1h.rs`

### 2.1 Constants

```rust
const SHIFT_A_PPM: f64 = 3.0;
const SHIFT_B_PPM: f64 = 3.5;
const J_HZ: f64 = 7.0;
```

Two protons 0.5 ppm apart with a 7 Hz coupling. 0.5 ppm at 600 MHz is 300 Hz, so $\rho = 7/300 \approx 0.023$ at our standard field — weak but visible. At 60 MHz the same 0.5 ppm is only 30 Hz, so $\rho = 7/30 \approx 0.23$: strong-coupling regime. That's the point of the scan.

### 2.2 Field scan

```rust
fn b0_for_mhz(mhz: f64) -> f64 {
    14.0954 * mhz / 600.0
}
```

14.0954 T is the canonical "600 MHz for 1H" field (it's where the Zeeman walkthrough's example lives). Any other spectrometer frequency is a linear rescale of $B_0$ — the gyromagnetic ratio is a nuclear constant, not a spectrometer knob.

```rust
for mhz in [60.0, 200.0, 600.0, 1000.0] {
    simulate_at(mhz);
}
```

Four fields, roughly bench-top NMR (60 MHz) through current state-of-the-art (1000 MHz). At each field we build the full pipeline and dump a CSV.

### 2.3 Building `H = H_Z + H_J`

```rust
let sys = SpinSystem::new(
    vec![
        Spin::labeled(Isotope::H1, SHIFT_A_PPM, "A"),
        Spin::labeled(Isotope::H1, SHIFT_B_PPM, "B"),
    ],
    b0,
);

let h_zeeman = ZeemanH::new(&sys);
let h_coupling = JCouplingH::new(&sys, &[(0, 1, J_HZ)]);
let h: Box<dyn Hamiltonian> =
    Box::new(SumH::new(vec![Box::new(h_zeeman), Box::new(h_coupling)]));
```

Three things to notice:

1. The Zeeman and J-coupling terms are constructed independently. Neither knows about the other. That's the trait paying off — new physics doesn't entangle old physics.
2. We `Box` each term into `Box<dyn Hamiltonian>` so `SumH` can hold a heterogeneous list. `Hamiltonian` is object-safe (no `Self` in method signatures, no generics in the trait definition), so this just works.
3. We type the result as `Box<dyn Hamiltonian>` to make it obvious to the reader that from here on down the pipeline, we're working through the trait, not against a concrete type.

### 2.4 Picking the propagator

```rust
let dt = 1.0 / 20_000.0;
let p = MatrixPropagator::new(h.as_ref(), dt);
```

We pick `MatrixPropagator` because we know `H_J` is non-diagonal. What happens internally:

1. `MatrixPropagator::new` calls `h.as_dense()`, which bottoms out in `SumH::as_dense`, which sums the children.
2. It scales by $-i\Delta t$ to form $A = -iH\Delta t$ (antihermitian for Hermitian $H$).
3. `A.exp()` does scaling-and-squaring Padé. The result is $U = \exp(-iH\Delta t)$, unitary to machine precision.
4. It caches $U^\dagger$ alongside $U$ so each FID step only has to do `U ρ U†` (a pair of matmuls), not recompute the exponential.

If we had picked `DiagonalPropagator::new` here, the call would panic with "DiagonalPropagator requires a Hamiltonian that advertises diagonal storage via `try_as_diagonal`". The panic is deliberate: it's telling you the trait bound you violated, in words.

If you want the code to pick automatically, wrap both in a factory that queries `try_as_diagonal`. Nothing stops you; the trait shape supports it.

### 2.5 The rest of the pipeline is unchanged

```rust
let rho0 = thermal_x_state(&sys);
let obs = total_m_minus(&sys);
let n_points = 16_384;
let fid = compute_fid(&p, &rho0, &obs, n_points);

let spectrum = DiscreteSpectrum::from_fid(&fid, dt);
```

Identical to the Zeeman example. `compute_fid` takes a `&dyn Propagator` — neither it nor `DiscreteSpectrum` knows or cares whether the propagator is diagonal or dense. If we add a `KrylovPropagator` later, or a `SparsePropagator`, these two lines don't change.

16 384 points at `dt = 50 μs` gives an acquisition window of ~0.82 s → frequency resolution $\Delta f = 1/T \approx 1.22$ Hz. That's comfortable for a $J = 7$ Hz splitting.

### 2.6 Reporting

```rust
let delta_ppm = (SHIFT_A_PPM - SHIFT_B_PPM).abs();
let delta_nu_hz = delta_ppm * mhz;
let rho = J_HZ / delta_nu_hz;

println!(
    "\n== {:>5.0} MHz ({:>7.4} T) — J/|Δν| = {J_HZ}/{:.1} = {:.4} ==",
    mhz, b0, delta_nu_hz, rho,
);
```

The $\rho$ column is the most important thing this example prints. Watching it shrink from ~0.23 at 60 MHz to ~0.014 at 1000 MHz is the physical story the example is telling.

---

## 3. Reading the spectra

Plot any of the output CSVs with:

```sh
python scripts/plot_spectrum.py ab_1h_60mhz_spectrum.csv
```

What you should see:

- **60 MHz**: strongly asymmetric four-line pattern. Inner lines near 3.2–3.3 ppm dominate; outer lines are small. This is "roofing": the inner pair of an AB quartet is always the larger pair, and the effect gets more dramatic as $\rho$ grows.
- **200 MHz**: still asymmetric but less so. You can start to convince yourself you're looking at two doublets.
- **600 MHz**: essentially two equal-intensity doublets at 3.0 and 3.5 ppm. Classic first-order appearance.
- **1000 MHz**: clean first-order doublets; the inner lines and outer lines are within ~1% of equal height.

Compare any of these to what a first-order simulator would predict — four lines at $\nu_A \pm J/2$ and $\nu_B \pm J/2$, all equal intensity. At 60 MHz that prediction is simply *wrong*; at 1000 MHz it's almost right.

---

## 4. Validating the physics

Two tests in the repo backstop this example:

### 4.1 `two_spin_strong_coupling_eigenvalues_match_analytic_ab`

(Unit test in `hamiltonian.rs`.) Builds $H_Z + H_J$ for a two-proton system, symmetric-eigendecomposes the dense matrix, and compares the four eigenvalues (in Hz) to the closed-form AB prediction:

$$
E_{\alpha\alpha} = \tfrac{1}{2}(\nu_A + \nu_B) + \tfrac{J}{4}, \quad
E_{\pm} = -\tfrac{J}{4} \pm \tfrac{1}{2}\sqrt{(\nu_A - \nu_B)^2 + J^2}, \quad
E_{\beta\beta} = -\tfrac{1}{2}(\nu_A + \nu_B) + \tfrac{J}{4}.
$$

If this test passes, the Hamiltonian matrix elements (Kronecker ordering, $2\pi J$ scaling, flip-flop sign) are all correct.

### 4.2 `weak_coupling_limit_two_spins_matches_first_order_doublets`

(Integration test in `tests/backend_agreement.rs`.) Takes two protons at 7.0 and 1.0 ppm with $J = 7$ Hz ($\rho \approx 0.004$ at 600 MHz, deeply first-order), runs the full quantum pipeline, and checks:

1. The four observed peak positions match $\nu_A \pm J/2$ and $\nu_B \pm J/2$ within one FFT bin.
2. The four peaks are equal-intensity (no roofing) within 3%.
3. The first-order backend produces the same pattern within the same tolerances.

If both tests pass, the quantum engine is correct at the two ends of the $\rho$ axis — and since the matrix is built from first principles (not interpolated), it's correct in between.

---

## 5. Design themes M3 reinforced

- **Trait-default `try_as_diagonal` is a negotiation, not a requirement.** `JCouplingH` opts *out* of the diagonal fast path; the trait default does all the work. If we add a `SparseJCouplingH` later, it can opt into a `try_as_sparse` fast path the same way.
- **Combinators compose, concrete types don't.** We didn't bake Zeeman + J into a single struct, even though "Zeeman + J" is the most common Hamiltonian in NMR. `SumH` lets that arbitrary sum live at the call site, where the user actually decides what physics they want in their simulation.
- **The example hasn't changed shape from Zeeman to AB.** Same seven steps, different propagator + different Hamiltonian. If we add M4 (RF pulses) or relaxation, examples that use them will still have this structure.
