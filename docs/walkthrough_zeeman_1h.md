# A guided tour of the `nmr_sim` quantum engine

**Audience:** you, six months from now, or anyone else wanting to understand how a single line of the `zeeman_1h` example actually produces a peak on a ppm axis.

This is a reading companion, not an API reference. It walks through every module in the quantum-engine stack in the order they get used, explains both the physics and the Rust choices, and then finishes with a line-by-line pass through `examples/zeeman_1h.rs`.

---

## 0. What the quantum engine does, in one paragraph

An NMR experiment takes a sample, drops it in a strong magnet, zaps the spins with a short RF pulse, and records the voltage that the precessing magnetization induces in a coil for the next second or so. That voltage trace is the **free induction decay** (FID). Fourier-transforming the FID gives the **spectrum** — peaks at each spin's chemical-shift frequency, with intensities proportional to how many spins are at that shift. The quantum engine in `nmr_sim` is a numerical model of that entire pipeline: it builds a quantum description of the molecule's nuclear spins, evolves that description in time under a Hamiltonian you specify, reads out a detection observable at every time step to form a simulated FID, and then FFTs that FID into a spectrum on a ppm axis.

The pipeline as implemented reads:

```
SpinSystem  →  ZeemanH  →  DiagonalPropagator  →  compute_fid  →  DiscreteSpectrum
 (who/where)    (energy)    (time evolution)     (detection)     (frequency view)
```

Every one of those arrows corresponds to one module of code. The rest of this doc walks through them in order.

---

## 1. The physics frame we're working in

Three quick ideas that everything downstream depends on.

### 1.1 The rotating frame

A 1H nucleus sitting in a 14.0954 T magnet precesses at ~600 MHz. If you tried to simulate that directly, every numerical step would have to resolve that oscillation, and all the interesting chemistry — chemical shifts of a few hundred Hz spread across ~10 ppm, J-couplings of a few Hz — would disappear into the roundoff of a vastly larger number. The standard fix is to jump into a frame rotating at the bare-isotope Larmor frequency. In that frame the 600 MHz precession has been subtracted off, and what's left is just the **chemical-shift offset**:

$$
\Delta\omega_i = -\gamma_i \cdot B_0 \cdot \delta_i \cdot 10^{-6}
$$

where $\gamma_i$ is the gyromagnetic ratio (rad·s⁻¹·T⁻¹), $B_0$ is the field (Tesla), and $\delta_i$ is the chemical shift in ppm. At 14.0954 T, 1 ppm of 1H corresponds to ~600 Hz (or $\approx 3770$ rad/s). This is the scale our Hamiltonians live at, and it's comfortable for f64 arithmetic. See `spin::Spin::shift_angular` — that's the function computing this number.

### 1.2 The density matrix

A single spin-1/2 has a 2-dimensional Hilbert space, so its state is a 2×2 Hermitian positive-semidefinite matrix with trace 1 — the **density matrix** $\rho$. For $N$ spin-1/2's, the state lives in a Hilbert space of dimension $2^N$, and $\rho$ is a $2^N \times 2^N$ matrix. Three spins: 8×8. Ten spins: 1024×1024. This is the exponential wall that all quantum-spin simulators run into, and it's why the architecture doc flags reduced-basis and polyadic-storage schemes as future work.

We actually track the **deviation density matrix** — $\rho$ with its uniform $\mathbb{1}/D$ piece subtracted off. That part commutes with every Hamiltonian, so it's stationary and contributes nothing to any traceless observable (and $M^-$, our detection operator, is traceless). Dropping it also eliminates a giant constant that would dominate $\rho$'s Frobenius norm. The only thing we lose is the absolute signal scale, which NMR spectra don't care about anyway. See the preamble of `src/state.rs` for the justification in narrative form.

### 1.3 The Liouville–von Neumann equation

Closed quantum systems evolve under

$$
\frac{d\rho}{dt} = -i[H, \rho] \quad\Longrightarrow\quad \rho(t + \Delta t) = U\,\rho(t)\,U^\dagger
$$

where $U = \exp(-iH\Delta t)$ is unitary. That second form is what `Propagator::apply` implements. The continuous differential equation becomes a discrete one-time-step-at-a-time marching algorithm.

Detection is

$$
s(t) = \mathrm{Tr}[\hat O \cdot \rho(t)]
$$

for some observable $\hat O$. For a standard 1D FID the conventional choice is $\hat O = M^- = \sum_i \hat I_{-,i}$; explained in the operator section.

With those three ideas in hand, the rest of the code is just bookkeeping around them.

---

## 2. `spin.rs` — who's in the system

This module answers *what nuclei are where*. It deliberately doesn't know anything about matrices; that's `operator.rs`'s job. `spin.rs` just describes the physical players.

### 2.1 `Isotope`

```rust
pub enum Isotope {
    H1, H2, C13, N15, F19, P31,
}
```

A plain enum with a fixed set of variants. Each knows:

- `two_i() -> u32` — twice the spin quantum number, as an integer. 1H, 13C, 19F, 31P are all spin-1/2 so `two_i = 1`. Deuterium is spin-1 so `two_i = 2`. Storing `2I` rather than `I` avoids float-equality headaches and makes the Hilbert-space dimension $2I + 1$ exact. Nuclei with higher spin (17O, 14N, etc.) slot in the same way.
- `multiplicity() -> u32` — returns `two_i() + 1`, the dimension of the single-spin Hilbert space.
- `gamma() -> f64` — the gyromagnetic ratio. **This can be negative** (15N). All the sign logic downstream depends on respecting that: if you ever short-circuit with `|gamma|`, you'll silently flip 15N peaks to the wrong side of the spectrum.
- `larmor_hz(b0)` — the signed Larmor frequency $\nu_0 = -\gamma B_0 / 2\pi$.

The gyromagnetic ratios are hardcoded constants near the top of the file, pulled from IAEA/IUPAC tables. If someone wants more sig figs later it's a one-line change; we're well below any experimental precision that matters for simulation at this stage.

### 2.2 `Spin`

```rust
pub struct Spin {
    pub isotope: Isotope,
    pub shift_ppm: f64,
    pub label: Option<String>,
}
```

A single nuclear site: an isotope plus a chemical shift plus an optional human-readable label. The label is for debugging and display only; nothing in the physics consumes it. `Spin::new` creates an unlabeled spin, `Spin::labeled` takes a label (as anything that `Into<String>`, so `"methyl"` works).

The two important methods:

- `shift_angular(b0) -> f64` implements $\Delta\omega_i = -\gamma_i B_0 \delta_i \cdot 10^{-6}$. This is the number that goes on the diagonal of the rotating-frame Zeeman Hamiltonian.
- `larmor_angular(b0) -> f64` returns the full Larmor frequency including the bare-isotope Larmor, $\omega_0 = -\gamma B_0 (1 + \delta \cdot 10^{-6})$. We almost never use this — the rotating-frame convention means the bare part has been subtracted off — but it's there for sanity checks.

Note the Rust idiom: `Spin` is a plain data struct with public fields, and the methods are pure functions of those fields. No interior state, no builders. For a plain-old-data type like this, that's the cleanest shape.

### 2.3 `SpinSystem`

```rust
pub struct SpinSystem {
    pub spins: Vec<Spin>,
    pub b0_tesla: f64,
}
```

A collection of spins, plus the static field they sit in. Two methods worth knowing:

- `len()` — number of spins.
- `dim() -> u64` — the full Hilbert-space dimension, $\prod_i (2I_i + 1)$. For three 1H: $2 \times 2 \times 2 = 8$. For a 1H + 2H + 13C mix: $2 \times 3 \times 2 = 12$. `u64` stays exact up to ~30 spin-1/2's, which is well past where anyone could fit the dense matrix in RAM, so we never have to worry about overflow.

Note that `SpinSystem` doesn't own any matrices. It's purely declarative: "here are the spins, here's the field." The `operator` module turns this into the matrix world.

---

## 3. `operator.rs` — single-spin and multi-spin matrices

This is where the linear-algebra machinery lives. It's the module that hands every downstream piece — Hamiltonians, states, observables — the actual matrices they build from.

### 3.1 The type

```rust
pub type Operator = DMatrix<Complex<f64>>;
```

An `Operator` is a dynamically-sized dense complex matrix from `nalgebra`. Dense rather than sparse because at the sizes we currently simulate (≤ ~10 spins → ≤ 1024×1024), dense is simpler and faster than sparse bookkeeping. Statically-sized nalgebra matrices would be faster still for fixed small N, but spin-system size is a runtime parameter, so dynamic it is.

### 3.2 Single-spin operators

The four core single-spin operators for arbitrary spin-$I$ live here. They all take `two_i: u32` (= 2I) and return an $(2I+1) \times (2I+1)$ matrix.

```rust
pub fn iz(two_i: u32) -> Operator
pub fn iplus(two_i: u32) -> Operator
pub fn iminus(two_i: u32) -> Operator  // iplus().adjoint()
pub fn ix(two_i: u32) -> Operator       // (I+ + I-) / 2
pub fn iy(two_i: u32) -> Operator       // (I+ - I-) / (2i)
```

`iz` is built directly as a diagonal matrix with entries $m = I, I-1, \ldots, -I$ in order. For spin-1/2 that gives `diag(+1/2, -1/2)`. For spin-1, `diag(+1, 0, -1)`.

`iplus` is built from the raising-operator matrix element

$$
\langle I, m+1 | \hat I_+ | I, m \rangle = \sqrt{I(I+1) - m(m+1)}
$$

which puts a nonzero superdiagonal on the matrix. For spin-1/2 that's just `[[0, 1], [0, 0]]`; for spin-1 you get the familiar $\sqrt 2$'s on `(0,1)` and `(1,2)`. `iminus` is the Hermitian adjoint of `iplus`, i.e. the subdiagonal version. `ix` and `iy` are standard linear combinations.

**Basis convention you should memorize:** row/column 0 is $m = +I$ (spin up for spin-1/2), and indices increase as $m$ decreases. This matters the moment you look at a matrix and try to interpret an entry.

There's a nice test in this file — `casimir_i_squared_equals_i_i_plus_one_times_identity` — that checks the SU(2) Casimir invariant $\hat I_x^2 + \hat I_y^2 + \hat I_z^2 = I(I+1) \mathbb{1}$. If you ever break the single-spin operator construction, that test will catch it before anything else fires.

### 3.3 The `lift` function

Once you have several spins, each single-spin operator has to be embedded into the full product Hilbert space. $\hat I_{z,i}$ as a matrix on the full space is

$$
\hat I_{z,i} = \underbrace{\mathbb{1} \otimes \cdots \otimes \mathbb{1}}_{i-1 \text{ factors}} \otimes \hat I_z \otimes \underbrace{\mathbb{1} \otimes \cdots \otimes \mathbb{1}}_{N-i \text{ factors}}
$$

That's exactly what `lift` computes: take a list of per-site dimensions, start with a 1×1 identity, and Kronecker in one factor per site — the target site gets the actual operator, every other site gets an identity of the right dimension.

```rust
pub fn lift(dims: &[usize], op: &Operator, site: usize) -> Operator
```

The Kronecker-product ordering we use (`result = result.kronecker(&factor)`, folding left-to-right) matches `nalgebra`'s convention and gives **the first spin varies slowest** in the product basis. For two spin-1/2's that means:

```
index 0: |↑↑⟩
index 1: |↑↓⟩   ← second spin flips first
index 2: |↓↑⟩
index 3: |↓↓⟩
```

If you ever decide to inspect a 4×4 density matrix by eye, the `(1, 2)` element is the $|\uparrow\downarrow\rangle\langle\downarrow\uparrow|$ coherence — a zero-quantum term. The `iz_at_site_*` tests pin this convention down with explicit diagonals.

The `_at` variants — `iz_at`, `ix_at`, `iy_at`, `iplus_at`, `iminus_at` — each just gather `site_dims(sys)` and call `lift`. Nothing fancy; they're convenience wrappers.

### 3.4 `total_m_plus` and `total_m_minus`

```rust
pub fn total_m_minus(sys: &SpinSystem) -> Operator  // Σᵢ Î₋,i
pub fn total_m_plus(sys: &SpinSystem) -> Operator   // Σᵢ Î₊,i
```

$M^-$ is the standard complex-FID detection operator. The reason it's $M^-$ rather than, say, $M_x$ or $M^+$ comes from the sign convention. With $M^-$, a spin at offset $\Delta\omega$ contributes a signal oscillating as $e^{-i\Delta\omega t}$. Our FFT convention is "forward transform has $\exp(-2\pi i km/N)$", so that $e^{-i\Delta\omega t}$ shows up at **positive** frequency $\Delta\omega / 2\pi$. For 1H at positive ppm that means a positive Hz offset and — after the gyromagnetic sign-flipping in the ppm conversion — a positive ppm on the final axis. Everything is consistent if we don't change any of those conventions. If you ever see a spectrum come out mirror-flipped, there's a very good chance one of those signs has flipped somewhere downstream.

---

## 4. `hamiltonian.rs` — the energy operator

This module defines the abstract `Hamiltonian` trait and the one concrete implementation we currently have, `ZeemanH`.

### 4.1 The trait

```rust
pub trait Hamiltonian {
    fn dim(&self) -> usize;
    fn as_dense(&self) -> Operator;
    fn try_as_diagonal(&self) -> Option<&DVector<f64>> { None }
}
```

A Hamiltonian exposes two views of itself, always in units of rad/s. `as_dense` is the universal fallback — any implementor has to be able to materialize a dense $D\times D$ matrix, even if it stores something more compact internally. `try_as_diagonal` is an *optional advertisement*: "I'm stored diagonally; here's the real diagonal as a `DVector<f64>`". Propagators consume these views to specialize: `DiagonalPropagator` asks `try_as_diagonal` and refuses the job if the answer is `None`; `MatrixPropagator` always calls `as_dense`. Future specializations (sparse, matrix-free, restricted-basis) will be further optional views on the same trait.

### 4.2 `ZeemanH`

```rust
pub struct ZeemanH {
    diagonal: DVector<f64>,
}
```

The rotating-frame Zeeman Hamiltonian:

$$
\hat H_Z = \sum_i \Delta\omega_i \cdot \hat I_{z,i}
$$

Built in the constructor by looping over spins: for each spin, compute $\Delta\omega_i = \text{spin.shift\_angular(b0)}$, lift $\hat I_z$ into the full space with `iz_at` (which materializes a $D\times D$ temporary), extract its diagonal, scale by $\Delta\omega_i$, and accumulate into `diagonal`. The loop skips spins at zero shift — their contribution is exactly zero and we save the Kronecker-product work.

Because every term in the sum is diagonal in the product-Iz basis (which is the basis `operator.rs` builds things in), `ZeemanH` is always diagonal. The test `zeeman_two_proton_diagonal_matches_hand_computation` writes out the expected four diagonal entries for a two-proton system explicitly — a good place to build intuition for the tensor-product basis.

**Design note (M2b):** storage is a real `DVector<f64>` of length $D$, not a $D\times D$ complex matrix. For an N-spin-1/2 system this is $2^N$ reals vs. $4^N$ complex numbers — a factor of $2^{N+1}$ memory win that becomes decisive as N grows (20 protons: 8 MB vs. 16 TB). `as_dense()` materializes the full matrix on demand for callers that want it; `try_as_diagonal()` returns `Some(&self.diagonal)` so `DiagonalPropagator` can skip the materialization entirely. When M3 brings J-coupling, `JCouplingH` will be stored densely (or, later, sparsely) and its `try_as_diagonal` will correctly return `None`, routing callers to `MatrixPropagator`.

---

## 5. `state.rs` — the initial density matrix

This module builds the $\rho(0)$ that an NMR experiment actually starts with. It's a short module — one function — but the function hides an instructive physics derivation.

### 5.1 The type alias

```rust
pub type DensityMatrix = Operator;
```

A `DensityMatrix` is just an `Operator` — the alias exists to document intent at type signatures. A function taking `&DensityMatrix` is asking for $\rho$, not an arbitrary operator. The type system doesn't enforce validity (Hermiticity, positive-semidefiniteness, trace); producers are responsible for building valid states.

### 5.2 `thermal_x_state`

```rust
pub fn thermal_x_state(sys: &SpinSystem) -> DensityMatrix {
    let dim = sys.dim() as usize;
    let mut rho = DensityMatrix::zeros(dim, dim);
    for (i, spin) in sys.spins.iter().enumerate() {
        let gamma = spin.isotope.gamma();
        let ix_i = ix_at(sys, i);
        rho += ix_i.map(|x| x * Complex::new(gamma, 0.0));
    }
    rho
}
```

Returns $\rho(0) = \sum_i \gamma_i \cdot \hat I_{x,i}$.

Where this formula comes from: the thermal-equilibrium density matrix in the high-temperature limit is

$$
\rho_{\text{eq}} \approx \frac{\mathbb{1}}{D} + \frac{\hbar B_0}{k_B T} \sum_i \gamma_i \, \hat I_{z,i}
$$

A perfect π/2-y pulse rotates $\hat I_z \to \hat I_x$, so post-pulse

$$
\rho_{\text{post-pulse}} \approx \frac{\mathbb{1}}{D} + \frac{\hbar B_0}{k_B T} \sum_i \gamma_i \, \hat I_{x,i}
$$

Drop the identity piece (it's stationary and contributes nothing to $M^-$), drop the overall $\hbar B_0 / k_B T$ prefactor (it only sets absolute signal size, which NMR spectra don't report), and you're left with $\sum_i \gamma_i \hat I_{x,i}$.

**Why the γ weighting is load-bearing:** temperature drops out of every relative intensity (because it's a global prefactor), but γ does not. A heteronuclear system — say 1H + 13C — correctly shows 13C peaks about ¼ the height of 1H peaks when you weight by γ, because $\gamma_C / \gamma_H \approx 0.25$. The test `heteronuclear_amplitudes_are_gamma_weighted` pins this ratio down directly.

The physical units of the returned matrix are "rad·s⁻¹·T⁻¹" (just γ's units, since we're dropping $B_0 \hbar / k_B T$). Numerically that means matrix entries on the order of $10^8$, which is fine for f64 and transparent after the FFT, which is homogeneous in its input.

---

## 6. `propagator.rs` — how time moves

### 6.1 The trait

```rust
pub trait Propagator {
    fn apply(&self, rho: &DensityMatrix) -> DensityMatrix;
    fn dt(&self) -> f64;
}
```

A propagator advances $\rho$ by one time step $\Delta t$ via $\rho \mapsto U \rho U^\dagger$. The trait exposes two things:

- `apply` — do one step.
- `dt()` — tell the caller what step size the propagator uses, so downstream code (the FID, specifically) can build the correct time axis.

Why a trait rather than a single concrete type? Because different regimes call for different propagation strategies. `DiagonalPropagator` is the cheapest when it applies. Once J-couplings come in (M3), `H` stops being diagonal and we'll need an `EigenbasisPropagator` (diagonalize once up-front, propagate in the eigenbasis) or a `KrylovPropagator` (iterative $\exp(-iH\Delta t) \cdot \rho$ for huge systems where diagonalizing is prohibitive). `compute_fid` takes any `P: Propagator`, so adding a new strategy will plug in without changing the FID code.

### 6.2 `DiagonalPropagator`

The specialization for when $H$ is diagonal. The math:

If $H = \mathrm{diag}(h_0, h_1, \ldots)$, then $U = \mathrm{diag}(e^{-ih_0\Delta t}, e^{-ih_1\Delta t}, \ldots)$, and for any matrix $\rho$,

$$
(U \rho U^\dagger)_{ij} = e^{-i(h_i - h_j)\Delta t} \cdot \rho_{ij}
$$

So propagation is pure elementwise phase multiplication: $O(D^2)$ per step, zero matrix multiplications. For a rotating-frame Zeeman Hamiltonian that's the cheapest propagator you can write.

The constructor:

1. Asserts the input matrix is square.
2. Verifies diagonality (all off-diagonals have magnitude below `1e-10`). If this trips, the caller is trying to propagate something that isn't diagonal — typically a J-coupled Hamiltonian — and they need a different propagator. The threshold is tight enough to catch a typical 7 Hz J-coupling, which would put off-diagonal terms of order $2\pi \cdot 7 \approx 44$ rad/s — six orders of magnitude above 1e-10.
3. Reads the real parts of the diagonal (the imaginary parts should be roundoff if $H$ is Hermitian and diagonal, and are discarded) and stores the precomputed phases `phases[k] = exp(-i * h_k * dt)` as a `DVector`.

`apply` then does a straightforward double loop: `result[(i, j)] *= phases[i] * phases[j].conj()`. That's the whole engine.

**Nyquist:** `DiagonalPropagator` doesn't police the time step. If you give it a $\Delta t$ larger than $\|H\|_2^{-1}$, you get an alias. At 14 T with ~12 ppm of 1H spread, $\|H\|_2 \approx 2\pi \cdot 7200 \approx 4.5 \times 10^4$ rad/s, so $\Delta t < 22\,\mu\text{s}$ is safe. Our example uses 50 µs which is comfortable as long as shifts stay within ~10 ppm — see the choice of `dt = 1 / 20_000` in the example and the comment next to it.

---

## 7. `fid.rs` — computing the time-domain signal

### 7.1 `trace_product`

```rust
pub fn trace_product(a: &Operator, b: &Operator) -> Complex<f64>
```

Computes $\mathrm{Tr}[A \cdot B]$ without forming $A \cdot B$. The identity

$$
\mathrm{Tr}[AB] = \sum_{i,j} A_{ij} B_{ji}
$$

drops the cost from $O(D^3)$ (dense matmul + trace) to $O(D^2)$ (elementwise sum). Crucial because `compute_fid` evaluates this trace at every FID sample — ~8k time points per simulation — and $D^3$ vs $D^2$ is the difference between "fast" and "unacceptable" at even modest system sizes.

### 7.2 `compute_fid`

```rust
pub fn compute_fid<P: Propagator>(
    propagator: &P,
    rho_initial: &DensityMatrix,
    observable: &Operator,
    n_points: usize,
) -> Vec<Complex<f64>>
```

The FID-generating loop:

```rust
let mut fid = Vec::with_capacity(n_points);
let mut rho = rho_initial.clone();
for _ in 0..n_points {
    fid.push(trace_product(observable, &rho));
    rho = propagator.apply(&rho);
}
```

Read this slowly. It is:

1. Start from the initial $\rho$.
2. Take a measurement: `fid[k] = Tr[observable · ρ(k·Δt)]`.
3. Advance $\rho$ by one time step.
4. Repeat.

The first sample is recorded *before* any propagation, so the returned `fid[k]` corresponds to time $t_k = k \cdot \text{propagator.dt()}$. This matters for the FFT axis — if you shift the sampling by half a step, peak positions move by half a bin.

The function is generic over `P: Propagator`. Any propagator works — you want a `KrylovPropagator` in M3 instead? Just pass it. `compute_fid` neither knows nor cares how `apply` is implemented.

### 7.3 The trace factor gotcha

The comment inside the two-spin FID test is worth internalizing. For a system with $N$ spin-1/2 particles,

$$
\mathrm{Tr}[\hat I_{x,i}^2] = 2^{N-2}
$$

where the trace is in the full $2^N$-dimensional Hilbert space. This comes from the tensor structure: $\hat I_{x,i}^2 = \hat I_x^2 \otimes \mathbb{1} \otimes \cdots$, so the trace factorizes as $\mathrm{Tr}[\hat I_x^2] \cdot \mathrm{Tr}[\mathbb{1}]^{N-1} = (1/2) \cdot 2^{N-1}$. For $N=1$ this is the textbook $1/2$; for $N=2$ it's $1$; for $N=3$ it's $2$. The two-spin FID test had to use $\gamma$ (not $\gamma/2$) as the per-spin amplitude because of this factor. If you ever write a test that computes a trace-over-lifted-operators closed form, remember to carry the $\mathrm{Tr}[\mathbb{1}_{\text{others}}]$ factor.

---

## 8. `discrete_spectrum.rs` — FFT and the ppm axis

### 8.1 The struct

```rust
pub struct DiscreteSpectrum {
    pub frequencies_hz: Vec<f64>,
    pub amplitudes: Vec<Complex<f64>>,
}
```

The frequency axis (strictly ascending, in Hz) and the complex FFT amplitudes at each bin. Both arrays are fftshifted: after construction, bin 0 is the most negative frequency, bin $N-1$ is the most positive, and DC sits somewhere in the middle (exactly at $\lfloor N/2 \rfloor$ for even $N$).

### 8.2 `from_fid`

```rust
pub fn from_fid(fid: &[Complex<f64>], dt: f64) -> Self
```

The heart of the module. What it does in sequence:

1. Clone the FID into a mutable buffer (rustfft is in-place).
2. Build an `FftPlanner<f64>` and ask for a forward FFT of length $N$.
3. Run the FFT: `fft.process(&mut buffer)`.
4. Build the "natural" (unshifted) frequency axis using the numpy-`fftfreq` convention:
   - Bins $[0, \lceil N/2 \rceil)$ map to frequencies $[0, f_{\text{Nyq}})$.
   - Bins $[\lceil N/2 \rceil, N)$ map to negative frequencies $[-f_{\text{Nyq}}, 0)$.
5. `rotate_left(split)` on both the frequency array and the FFT buffer. This is fftshift: it swaps the two halves so the axis becomes strictly ascending and the spectrum displays with DC in the middle.

The even-vs-odd $N$ handling is done with `n.div_ceil(2)`. For even $N$ (the common case), `split = N/2`, and bin $N/2$ — the Nyquist bin — is assigned to the negative side by convention. For odd $N$ (never actually used in the example but supported), the math works out cleanly because `div_ceil` gives $\lceil N/2 \rceil$.

**Sign convention:** rustfft's forward transform is $S[m] = \sum_k s[k] e^{-2\pi i km/N}$. A time-domain $e^{+i\Omega t}$ maps to a positive-frequency peak at $\Omega/2\pi$. Combined with our choice of $M^-$ as the detection operator (which makes each spin oscillate as $e^{-i\Delta\omega t}$), a chemical-shift-offset peak lands at $-\Delta\omega/2\pi$ in Hz. After the ppm conversion below, that lands at positive $\delta$ for 1H.

### 8.3 Hz → ppm

```rust
pub fn frequencies_ppm(&self, reference_isotope: Isotope, b0_tesla: f64) -> Vec<f64>
```

Derivation (the docstring spells this out; I'm repeating it here so you can read it once in a less condensed form):

Our Hamiltonian gives $\Delta\omega = -\gamma B_0 \delta \cdot 10^{-6}$.
A spin at offset $\delta$ contributes a FID $\propto e^{-i\Delta\omega t}$.
The FFT puts that peak at Hz frequency $f = -\Delta\omega / (2\pi)$.
Substituting: $f = \gamma B_0 \delta \cdot 10^{-6} / (2\pi)$, so $\delta = 2\pi \cdot 10^6 \cdot f / (\gamma B_0)$.

That's the `hz_to_ppm` factor in the code. For 1H (γ > 0) it maps positive Hz → positive ppm. For 15N (γ < 0) it inverts the sign — correct, because 15N's negative γ really does flip the Larmor precession direction.

Note the function takes a `reference_isotope` argument. For homonuclear experiments you pass the one isotope you're simulating. For heteronuclear experiments you pick which channel you want to display — there isn't one "true" ppm axis, because each isotope has its own rotating frame and its own γ.

### 8.4 Magnitude, real, imag, CSV

These are one-liners:

- `magnitude()` returns $|S[m]|$ at each bin — the simplest, phase-free display. Broader peaks than absorption-mode (√2× FWHM for a Lorentzian).
- `real()` and `imag()` expose the absorption/dispersion split. Without phase correction they're whatever the raw FFT gave; phase correction and auto-phasing is future work.
- `to_csv_ppm(filename, isotope, b0)` writes the **absorption-mode** spectrum (real part of the FFT) vs ppm to a CSV, sorted by descending ppm so the file plots directly in NMR-standard orientation (downfield on the left). Absorption is what Mestrenova/Topspin display after phase correction; our standard pipeline is phased by construction (real thermal state + $M^-$ observable → $\mathrm{FID}(0)$ is real positive), so no phase correction is needed. Use `magnitude()` + your own writer if you ever need the phase-free display instead.

---

## 9. `examples/zeeman_1h.rs` — putting it all together

Here's the example in full, with walking commentary.

```rust
use nmr_sim::operator::total_m_minus;
use nmr_sim::{
    compute_fid, thermal_x_state, DiagonalPropagator, DiscreteSpectrum, Isotope, Spin, SpinSystem,
    ZeemanH,
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
```

Three 1H spins at 1.5, 3.7, and 7.2 ppm in a 14.0954 T field. The labels are just decoration; the physics uses only isotope and shift. `sys.dim()` prints 8 because $2 \times 2 \times 2 = 8$.

The ppms are roughly representative: 1.5 could be a methyl, 3.7 a methine near an oxygen, 7.2 an aromatic proton. Nothing in the simulator cares — these are just three independent chemical shifts for a singlets-only demo.

```rust
    // --- 2. Hamiltonian ---
    let h = ZeemanH::new(&sys);
```

Builds the 8×8 rotating-frame Zeeman matrix. Only the diagonal is nonzero. For our three shifts, the three $\Delta\omega$'s are approximately $(-5.65, -13.94, -27.12) \times 10^3$ rad/s, or about $(-900, -2220, -4315)$ Hz. The diagonal of the 8×8 matrix is every sign-combination of $\{\pm \Delta\omega_1/2, \pm \Delta\omega_2/2, \pm \Delta\omega_3/2\}$ — 8 entries total, following the first-spin-slowest tensor basis ordering.

```rust
    // --- 3. Propagator ---
    let dt = 1.0 / 20_000.0;
    let p = DiagonalPropagator::new(&h, dt);
```

20 kHz sampling → Nyquist at 10 kHz, which comfortably covers chemical shifts up to ~15 ppm at 600 MHz (9 kHz offsets). The constructor calls `h.try_as_diagonal()`, which `ZeemanH` answers with `Some(&self.diagonal)`. From that 8-element real vector we precompute the 8 per-basis-state phase factors $e^{-i h_k \Delta t}$. All subsequent `p.apply` calls are pure elementwise multiplication — no D×D matrix ever materialized. (If we'd used a non-diagonal Hamiltonian like a future `JCouplingH`, `try_as_diagonal` would return `None` and `DiagonalPropagator::new` would panic, directing us to `MatrixPropagator` instead.)

```rust
    // --- 4. FID ---
    let n_points = 8192;
    let rho0 = thermal_x_state(&sys);
    let observable = total_m_minus(&sys);
    let fid = compute_fid(&p, &rho0, &observable, n_points);
```

The FID pipeline:

- `rho0 = γ_H · Σᵢ Îx,i` — built by `thermal_x_state`. An 8×8 matrix.
- `observable = M⁻ = Σᵢ Î₋,i` — built by `total_m_minus`. Another 8×8 matrix.
- `fid` is a `Vec<Complex<f64>>` of length 8192. Each entry is $s[k] = \mathrm{Tr}[M^- \cdot \rho(k \Delta t)]$.

The total acquisition time is $N \cdot \Delta t = 8192 / 20000 = 0.41$ s. That's short compared to typical NMR T₂ (which we're not modeling yet anyway), but long enough that the frequency resolution $\Delta f = 1/(N \Delta t) \approx 2.44$ Hz is comfortably finer than the chemical-shift spacings.

Inside the loop, each time step is ~$D^2 = 64$ phase multiplications for the propagator plus a $D^2 = 64$-term trace. Trivially fast at this system size — most of `cargo run --release` time is in cargo overhead, not the simulation.

```rust
    // --- 5. Spectrum ---
    let spectrum = DiscreteSpectrum::from_fid(&fid, dt);
    println!(
        "Spectrum: {} bins, Δf = {:.2} Hz ({:.4} ppm)",
        spectrum.len(),
        spectrum.freq_resolution_hz(),
        spectrum.freq_resolution_hz() * 2.0 * std::f64::consts::PI * 1e6
            / (Isotope::H1.gamma() * b0),
    );
```

FFT time. `from_fid` runs rustfft on the 8192-point FID, builds the frequency axis in Hz, and fftshifts both. The print statement converts Δf (2.44 Hz) to ppm (≈ 0.004 ppm at 600 MHz) using the same formula `frequencies_ppm` uses internally.

```rust
    let ppms = spectrum.frequencies_ppm(Isotope::H1, b0);
    let mags = spectrum.magnitude();
    let mut indexed: Vec<(usize, f64)> = mags.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top 3 peaks:");
    for (i, (idx, mag)) in indexed.iter().take(3).enumerate() {
        println!("  #{}: {:.3} ppm, mag = {:.3e}", i + 1, ppms[*idx], mag);
    }
```

Find the three loudest bins and print their ppm and magnitude. In practice you get:

```
#1: 7.200 ppm, mag = 4.319e12
#2: 1.501 ppm, mag = 3.874e12
#3: 3.702 ppm, mag = 2.979e12
```

The ppms match the input shifts to within one bin ($\pm$0.002 ppm). The magnitudes are not all equal — they should be identical in principle (each 1H contributes equally, and all shifts have the same proton count) but the sinc interpolation at different fractional bin offsets attenuates each peak differently. This is the exact artifact that bit the backend-agreement test: unapodized quantum FFT peaks have $\mathrm{sinc}(\Delta)$-dependent peak heights where Δ is the peak's fractional-bin offset. For a "real" NMR-looking spectrum we'd apodize the FID (an exponential decay window → Lorentzian convolution) and the sinc artifact would vanish. That's a minor future improvement; the peak positions are already correct, and the L2 norm of each peak (the Parseval-invariant quantity we use in the backend test) already agrees across backends.

```rust
    // --- 6. CSV output ---
    match spectrum.to_csv_ppm("zeeman_1h_spectrum.csv", Isotope::H1, b0) {
        Ok(()) => println!("Saved zeeman_1h_spectrum.csv"),
        Err(e) => eprintln!("Error writing CSV: {e}"),
    }
}
```

Dump the absorption-mode (real-part) spectrum vs ppm to CSV, sorted descending in ppm (NMR-standard orientation). Open it in any plotting tool and you'll see three narrow, sinc-shaped peaks at 1.5, 3.7, and 7.2 ppm — or, with apodization + zero-fill applied, smooth Lorentzians at those positions.

---

## 10. Design themes to notice

A few recurring patterns in how the engine is put together, worth naming so you can spot them when you're extending the code:

### 10.1 Units discipline

Everything internal is in SI (or its NMR cousins): chemical shifts in ppm, fields in Tesla, gyromagnetic ratios in rad·s⁻¹·T⁻¹, Hamiltonians in rad/s, time in seconds. Ppm and Hz are only for presentation — the conversions happen at I/O boundaries (`shift_angular` into rad/s on the way in, `frequencies_ppm` on the way out). If you always ask "what units is this number in?", the formulas stay simple and the sign bookkeeping stays tractable.

### 10.2 Rotating-frame convention

Every Hamiltonian we write is in the rotating frame at the bare-isotope Larmor. This is not optional; working in the lab frame would make the numerics hopeless. The convention is documented in the `hamiltonian.rs` preamble and in `docs/architecture.md`. When you add J-coupling (M3) or RF pulses (M4+), make sure the rotating-frame transformation is still respected — the secular approximation and the shift-frame choices will become load-bearing.

### 10.3 Traits for propagation strategies

`Propagator` is deliberately trait-based, so we can swap in `EigenbasisPropagator`, `KrylovPropagator`, or `SparsePropagator` without touching `compute_fid`. The M2b refactor will add a non-diagonal propagator, and the M3 Hamiltonian (J-coupling) will force its use. `Hamiltonian` is similarly a trait, so Zeeman can coexist with J-coupling and (eventually) dipolar terms, each producing their own matrix via the same interface.

### 10.4 Hermiticity and the sanity checks

Several tests verify Hermiticity, tracelessness, or unitarity after evolution. These aren't paranoia: they're load-bearing invariants of the physics, and a bug in operator construction or propagation will trip them long before it garbles a spectrum in a way you can visually spot. The propagator-preserves-hermiticity test is a particularly useful one — if it ever fires, you've broken $U\rho U^\dagger$ preservation at the matrix level.

### 10.5 The first-spin-varies-slowest basis

Every multi-spin matrix in this crate uses the Kronecker-product convention where spin 0 varies slowest. The `iz_at_site_zero_two_protons` and `iz_at_site_one_two_protons` tests make this explicit. If you ever introduce code that ingests matrices from elsewhere (e.g., a Python reference implementation with different ordering), the Kronecker convention is the first place to check.

### 10.6 The exponential wall is not far away

Dense $D \times D$ matrices work fine up to ~10 spin-1/2's (1024×1024 is ~16 MB for complex f64). Twelve spins is 4096×4096 ≈ 256 MB. Fourteen is 16384² ≈ 4 GB. Beyond that the dense representation stops fitting in memory on a laptop. The existing architecture makes that wall visible — every big allocation shows up in `operator::lift` and `total_m_minus` — and it's the reason the architecture doc flags polyadic decomposition and Liouville-space reduction for later milestones. Don't be seduced into writing code that assumes dense is forever.

---

## 11. What comes next

Both milestones that this doc originally flagged as "next" have now shipped:

- **M2b — `MatrixPropagator`.** Implemented. It's the universal `O(D^3)` apply / `O(D^2)` storage fallback built on `nalgebra`'s scaling-and-squaring Padé matrix exponential, and it kicks in automatically for any `Hamiltonian` that returns `None` from `try_as_diagonal`.

- **M3 — J-coupling.** Implemented via `JCouplingH` (full isotropic $2\pi J_{ij}\,\hat I_i\!\cdot\!\hat I_j$) and a `SumH` combinator so that `H = H_Z + H_J + \dots$ composes naturally through the `Hamiltonian` trait. Because `JCouplingH` doesn't advertise diagonal storage, any `SumH` that contains one routes through `MatrixPropagator` automatically — no caller-side changes.

See [`walkthrough_ab_system_1h.md`](walkthrough_ab_system_1h.md) for the M3 companion walkthrough: the same pipeline below `SumH`, but with the non-diagonal $\hat I_i\!\cdot\!\hat I_j$ Hamiltonian producing a real AB quartet with roofing — and collapsing into first-order doublets as $B_0$ increases. Everything in the current doc above `ZeemanH` (`compute_fid`, `DiscreteSpectrum`, CSV output) is shared between the two examples and does not change. That's the point of the trait layering.

Roadmap past M3 lives in `docs/architecture.md`; the near-term menu includes sparse/Krylov/restricted-basis propagator specializations, relaxation (Redfield/Lindblad) as new `Hamiltonian`/superoperator terms, and heteronuclear secular handling in `JCouplingH`.
