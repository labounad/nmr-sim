//! Hamiltonians for spin dynamics.
//!
//! Provides the [`Hamiltonian`] trait and concrete implementations of the
//! Hamiltonian terms relevant to NMR. **All Hamiltonians are expressed in the
//! rotating frame (one frame per isotope) and in units of rad/s.** This is
//! not negotiable: working in the lab frame would put numerically enormous
//! Larmor frequencies on the diagonal and destroy the interesting physics to
//! roundoff. See `docs/architecture.md` for the convention.
//!
//! # The `Hamiltonian` trait shape
//!
//! The trait exposes three views of the underlying operator, from most
//! structured to least:
//!
//! - [`as_dense`](Hamiltonian::as_dense): a full `D×D` dense matrix.
//!   Always available; the universal fallback.
//! - [`try_as_diagonal`](Hamiltonian::try_as_diagonal): the diagonal of H as
//!   a real `DVector<f64>`, if the Hamiltonian is stored diagonally in the
//!   product-Iz basis. Returns `None` by default.
//! - [`try_as_sparse`](Hamiltonian::try_as_sparse): a borrowed real CSR view
//!   ([`SparseH`]), if the Hamiltonian is naturally sparse in the
//!   product-Iz basis. Returns `None` by default. This is the fast path
//!   for Krylov-style propagators.
//!
//! This split lets propagators specialize without paying for structure
//! detection on every construction. A
//! [`DiagonalPropagator`](crate::propagator::DiagonalPropagator) asks
//! `try_as_diagonal` and either gets the data it needs in O(D) memory or
//! refuses the job up-front. A matrix-free Krylov propagator asks
//! `try_as_sparse` and iterates the CSR structure. A general
//! [`MatrixPropagator`](crate::propagator::MatrixPropagator) always calls
//! `as_dense` and pays the O(D²) cost.
//!
//! # Current implementations
//!
//! - [`ZeemanH`]: isotropic chemical-shift (rotating-frame Zeeman) Hamiltonian.
//!   Diagonal in the product-Iz basis; stored as `DVector<f64>` plus a cached
//!   [`SparseH`] view.
//! - [`JCouplingH`]: isotropic scalar coupling Σᵢ<ⱼ 2π Jᵢⱼ Îᵢ·Îⱼ. Non-diagonal
//!   in the product-Iz basis but structurally very sparse (block-diagonal in
//!   total M_z). Stored as [`SparseH`]; `as_dense` rematerializes on demand.
//!   For a fully spin-½ homonuclear system the sparse is built directly from
//!   the `(i, j, J)` triples in O(N_pairs · D), bypassing any dense matmul.
//! - [`SumH`]: additive combinator over any sequence of [`Hamiltonian`] terms.
//!   The standard way to write `H = H_Z + H_J + …`. Advertises `try_as_diagonal`
//!   iff every child does (preserving the O(D) diagonal fast path through
//!   composition) and `try_as_sparse` iff every child does (preserving the
//!   sparse fast path through composition).
//!
//! # Planned next
//!
//! - `DipolarH`: direct dipolar coupling with orientation dependence (ssNMR).
//! - `RfPulseH`: time-dependent RF irradiation (requires extending the trait).
//! - Heteronuclear J: currently [`JCouplingH::new`] rejects heteronuclear
//!   pairs; secular-approximation handling (zz-only) will be added alongside
//!   a proper multi-frame representation.

use crate::operator::{ix_at, iy_at, iz_at, Operator};
use crate::spin::SpinSystem;
use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use num_complex::Complex;
use std::f64::consts::PI;

/// Real-valued sparse Hamiltonian representation.
///
/// Every Hamiltonian currently shipped by this crate ([`ZeemanH`],
/// [`JCouplingH`], and [`SumH`] compositions of them) is purely real in the
/// product-Iz basis — the Îy-Îy terms in Îᵢ·Îⱼ exactly cancel the imaginary
/// parts of Îx·Îx + Îy·Îy + Îz·Îz. Storing real-valued keeps memory and
/// sparse-matvec cost at half of the complex equivalent and enables reuse of
/// nalgebra's `SymmetricEigen` (as [`crate::EigenPropagator`] already does for
/// its dense path).
///
/// Complex-valued sparse Hamiltonians — e.g. rotating-frame off-resonance
/// RF pulses — will land as a sibling type and a distinct trait hook
/// (`try_as_sparse_complex`) rather than widening this one.
pub type SparseH = CsrMatrix<f64>;

/// A Hamiltonian over a finite-dimensional Hilbert space, in rad/s.
///
/// Implementors expose a universal dense view via [`as_dense`](Self::as_dense)
/// and may optionally advertise structured storage via
/// [`try_as_diagonal`](Self::try_as_diagonal). More structured views
/// (e.g. `try_as_sparse`, `try_as_kronecker`) may be added in future
/// milestones; each is an optional fast path that a propagator can opt into.
pub trait Hamiltonian {
    /// Hilbert-space dimension (D = ∏ᵢ (2Iᵢ + 1) for a product system).
    fn dim(&self) -> usize;

    /// A dense `D×D` matrix representation in rad/s.
    ///
    /// Returned by value because implementors that store their Hamiltonian in
    /// a structured form (diagonal, sparse, matrix-free, …) materialize the
    /// dense matrix on demand and do not retain it. Callers that need the
    /// dense matrix repeatedly should hold onto the returned value themselves.
    fn as_dense(&self) -> Operator;

    /// If this Hamiltonian is diagonal in its storage basis, return a
    /// borrowed view of the real diagonal `DVector<f64>`; otherwise `None`.
    ///
    /// The diagonal is real because any Hermitian matrix has real diagonal
    /// entries, and any *diagonally stored* Hamiltonian is necessarily
    /// Hermitian in that basis.
    ///
    /// Default implementation returns `None` — i.e. implementors must opt in
    /// to advertising diagonal storage. A `Some` return is a claim by the
    /// implementor that the operator is *exactly* diagonal, not merely
    /// approximately so.
    fn try_as_diagonal(&self) -> Option<&DVector<f64>> {
        None
    }

    /// If this Hamiltonian is naturally sparse in the product-Iz basis,
    /// return a borrowed view of its real CSR representation; otherwise
    /// `None`.
    ///
    /// This is the fast path that [`crate::propagator::KrylovPropagator`]
    /// (M4a) and any future matrix-free propagators will opt into. A `Some`
    /// return is a claim that the operator is real-valued (see [`SparseH`])
    /// and Hermitian as stored — callers may rely on both properties without
    /// defensive checks.
    ///
    /// Default implementation returns `None`. An implementor that knows its
    /// operator admits a compact sparse form in rad/s should cache that form
    /// alongside its dense/diagonal storage and return it here. Zero entries
    /// (e.g. from zero-valued J couplings) must be dropped before CSR
    /// construction — consumers use `nnz()` as a cost proxy.
    fn try_as_sparse(&self) -> Option<&SparseH> {
        None
    }
}

/// Rotating-frame Zeeman (chemical-shift) Hamiltonian.
///
/// ```text
/// Ĥ_Z = Σᵢ Δωᵢ · Îz,ᵢ        where Δωᵢ = −γᵢ · B₀ · δᵢ · 10⁻⁶
/// ```
///
/// The bare-isotope Larmor frequencies are absorbed into each isotope's
/// rotating-frame reference, leaving only the chemical-shift offsets Δωᵢ
/// on the diagonal. The result is diagonal in the product-Iz basis, so this
/// implementation stores only the diagonal as a `DVector<f64>` — `2^N` reals
/// for an N-spin-1/2 system, not `4^N` complex numbers.
///
/// For a homonuclear system, this is the standard "offsets only" Zeeman
/// term. For heteronuclear systems, each isotope is implicitly in its own
/// rotating frame — a design we'll make explicit when J/dipolar couplings
/// between heteronuclei are added (they bring in the secular approximation).
#[derive(Debug, Clone)]
pub struct ZeemanH {
    /// Real diagonal in rad/s.
    diagonal: DVector<f64>,
    /// Cached sparse view of the diagonal as a real CSR matrix — materialized
    /// once at construction for the sparse-aware fast path (Krylov propagator).
    /// Zero entries on the diagonal are dropped so `nnz()` reflects the true
    /// structural sparsity, which in turn lets `SumH::try_as_sparse` take
    /// advantage of it without double-filtering.
    sparse: SparseH,
}

impl ZeemanH {
    /// Build the rotating-frame Zeeman Hamiltonian for `sys`.
    ///
    /// Time cost is O(N · D²) because we still reuse the generic `iz_at`
    /// single-site lift to extract per-spin diagonals — this materializes a
    /// D×D temporary per spin but only the diagonal survives. A future
    /// specialization can compute each Îz,ᵢ diagonal directly in O(D)
    /// without the temporary; that optimization is deferred because it
    /// requires mixed-radix index bookkeeping and we want this refactor to
    /// be a behavior-preserving step.
    pub fn new(sys: &SpinSystem) -> Self {
        let dim = sys.dim() as usize;
        let mut diagonal = DVector::<f64>::zeros(dim);
        for (i, spin) in sys.spins.iter().enumerate() {
            let omega = spin.shift_angular(sys.b0_tesla);
            // Skip spins at zero shift — their contribution is exactly zero
            // and the allocation/kronecker work is wasted. Not correctness-
            // critical; purely a small constant-factor win for sparse systems.
            if omega == 0.0 {
                continue;
            }
            let iz_i = iz_at(sys, i);
            for k in 0..dim {
                diagonal[k] += omega * iz_i[(k, k)].re;
            }
        }
        let sparse = diagonal_to_sparse(&diagonal);
        Self { diagonal, sparse }
    }

    /// Borrow the real diagonal directly. Equivalent to
    /// `self.try_as_diagonal().unwrap()` but statically infallible.
    pub fn diagonal(&self) -> &DVector<f64> {
        &self.diagonal
    }
}

/// Build a square real CSR matrix whose nonzero entries are the nonzero
/// entries of `diag` placed on the main diagonal.
///
/// Factored out because the same pattern is needed by at least two call sites
/// (ZeemanH construction, and — once we materialize diagonal SumH cached
/// diagonals into sparse — potentially more). Zero entries are dropped; the
/// result's `nnz()` is the count of nonzeros in `diag`.
fn diagonal_to_sparse(diag: &DVector<f64>) -> SparseH {
    let n = diag.len();
    let mut coo = CooMatrix::<f64>::new(n, n);
    for (k, &v) in diag.iter().enumerate() {
        if v != 0.0 {
            coo.push(k, k, v);
        }
    }
    CsrMatrix::from(&coo)
}

/// Extract the real CSR view of a dense (nominally real) complex Hamiltonian.
///
/// # Tolerance
///
/// Both real-zero entries (structural sparsity of the Hamiltonian) and
/// imaginary residuals (which for the Hamiltonians in this crate are
/// mathematically zero) are dropped below `SPARSE_REAL_TOL`. The tolerance is
/// deliberately loose relative to roundoff (`1e-12`, compared to the `~1e-15`
/// you'd see in pure f64 arithmetic) so that a denser-than-expected
/// multi-Kronecker build (e.g. large N with accumulated products of
/// Îx·Îx + Îy·Îy + Îz·Îz) doesn't leak structural zeros into the sparse view.
///
/// # Panics
///
/// Panics if any matrix element has `|Im z| > SPARSE_REAL_TOL` — a real-sparse
/// view of a complex Hamiltonian is a violation of the [`SparseH`] contract
/// and would silently produce wrong physics downstream.
fn dense_to_real_sparse(m: &Operator) -> SparseH {
    const SPARSE_REAL_TOL: f64 = 1e-12;

    let (nrows, ncols) = m.shape();
    let mut coo = CooMatrix::<f64>::new(nrows, ncols);
    for j in 0..ncols {
        for i in 0..nrows {
            let z = m[(i, j)];
            assert!(
                z.im.abs() < SPARSE_REAL_TOL,
                "dense_to_real_sparse: entry ({i}, {j}) = {z:?} has imaginary \
                 magnitude {} > tol {SPARSE_REAL_TOL:.0e}; the sparse fast \
                 path assumes a real-valued Hamiltonian in the product-Iz \
                 basis.",
                z.im.abs(),
            );
            if z.re.abs() >= SPARSE_REAL_TOL {
                coo.push(i, j, z.re);
            }
        }
    }
    CsrMatrix::from(&coo)
}

impl Hamiltonian for ZeemanH {
    fn dim(&self) -> usize {
        self.diagonal.len()
    }

    fn as_dense(&self) -> Operator {
        let n = self.diagonal.len();
        let mut m = Operator::zeros(n, n);
        for k in 0..n {
            m[(k, k)] = Complex::new(self.diagonal[k], 0.0);
        }
        m
    }

    fn try_as_diagonal(&self) -> Option<&DVector<f64>> {
        Some(&self.diagonal)
    }

    fn try_as_sparse(&self) -> Option<&SparseH> {
        Some(&self.sparse)
    }
}

/// Isotropic scalar J-coupling Hamiltonian.
///
/// ```text
/// Ĥ_J = Σ_{i<j} 2π J_{ij} · Îᵢ·Îⱼ
///     = Σ_{i<j} 2π J_{ij} · (Îx,ᵢ Îx,ⱼ + Îy,ᵢ Îy,ⱼ + Îz,ᵢ Îz,ⱼ)
///     = Σ_{i<j} 2π J_{ij} · (Îz,ᵢ Îz,ⱼ + ½(Î⁺ᵢ Î⁻ⱼ + Î⁻ᵢ Î⁺ⱼ))
/// ```
///
/// The `Îz·Îz` piece is diagonal in the product-Iz basis; the ladder piece
/// (flip-flop) connects states with the same total M_z but swapped single-spin
/// projections (e.g. |αβ⟩ ↔ |βα⟩). Those off-diagonal terms are precisely
/// why strong-coupling NMR spectra differ from first-order spectra — and why
/// this type returns `None` from `try_as_diagonal` and gets routed through
/// [`MatrixPropagator`](crate::propagator::MatrixPropagator).
///
/// # Homonuclear only, for now
///
/// The constructor panics if asked to couple two spins of different isotopes.
/// Strong-coupling (full Î·Î) is only physical between spins in the *same*
/// rotating frame — for heteronuclear pairs the flip-flop terms oscillate at
/// ~|γ_A − γ_B|·B₀ and are dropped under the secular approximation, leaving
/// a zz-only coupling. That variant is a planned follow-up.
///
/// # Storage
///
/// Internal storage is a real CSR sparse matrix ([`SparseH`]) — O(nnz), not
/// O(D²). That's what makes JCouplingH viable at N ≥ 14 where a 4 GB+ dense
/// cache is the very thing we're trying to avoid. [`as_dense`](Self::as_dense)
/// materializes a fresh complex D×D matrix on demand from the sparse in
/// O(nnz) time.
///
/// # Construction cost
///
/// Two code paths, selected automatically from the spin system:
///
/// - **Spin-½ homonuclear fast path** (all spins in `sys` are I=½): direct
///   sparse construction in O(N_pairs · D) time and O(nnz) peak memory, via
///   bit-indexed enumeration of the product-Iz basis. No dense intermediate
///   is ever materialized. This is the path Lucas's 1H NMR workloads hit.
/// - **Higher-spin (mixed-radix) fallback** (any spin has I > ½ — currently
///   just 2H): build the full dense Îᵢ·Îⱼ via kronecker-lifted operators at
///   O(N_pairs · D³), then extract the sparse. Tolerable while we don't run
///   large higher-spin systems; a mixed-radix direct-sparse path is future
///   work behind the same API.
#[derive(Debug, Clone)]
pub struct JCouplingH {
    /// Hilbert-space dimension D of the underlying spin system. Cached so
    /// `dim()` is O(1) independent of the sparse layout.
    dim: usize,
    /// Real CSR sparse representation in rad/s. Every nonzero matrix element
    /// of a homonuclear isotropic J-coupling is real in the product-Iz basis
    /// (the Îy-Îy imaginary parts cancel), so we safely keep only the real
    /// component. This is both the primary storage and the view returned by
    /// [`try_as_sparse`](Hamiltonian::try_as_sparse).
    sparse: SparseH,
}

impl JCouplingH {
    /// Build the J-coupling Hamiltonian for `sys` given a list of
    /// `(spin_i, spin_j, J_hz)` triples.
    ///
    /// - Spin-index order within a triple does not matter: `(i, j, J)` and
    ///   `(j, i, J)` produce the same contribution.
    /// - Duplicate pairs are summed: listing `(0, 1, 5.0)` and `(1, 0, 3.0)`
    ///   gives an effective `J_{01}` of 8 Hz. This is usually a user error;
    ///   we don't reject it because the semantics are unambiguous.
    /// - Entries with `J_hz == 0.0` are skipped without allocating.
    ///
    /// # Panics
    ///
    /// - If any `spin_i` or `spin_j` is out of range for `sys`.
    /// - If `spin_i == spin_j` (self-coupling is not physical).
    /// - If the two coupled spins are different isotopes (see the type-level
    ///   docs for the heteronuclear story).
    pub fn new(sys: &SpinSystem, couplings: &[(usize, usize, f64)]) -> Self {
        // Validate every triple upfront. These invariants are independent of
        // the fast/slow dispatch below, so hoisting them keeps the construction
        // functions narrow and the error messages in one place.
        for &(i, j, _) in couplings {
            assert!(
                i < sys.len(),
                "JCouplingH: spin index i = {i} out of range (system has {} spins)",
                sys.len(),
            );
            assert!(
                j < sys.len(),
                "JCouplingH: spin index j = {j} out of range (system has {} spins)",
                sys.len(),
            );
            assert_ne!(
                i, j,
                "JCouplingH: self-coupling (i == j == {i}) is not physical"
            );
            assert_eq!(
                sys.spins[i].isotope, sys.spins[j].isotope,
                "JCouplingH currently rejects heteronuclear couplings \
                 ({i}: {}, {j}: {}). Homonuclear only until secular-approximation \
                 handling lands in a future milestone.",
                sys.spins[i].isotope, sys.spins[j].isotope,
            );
        }

        let dim = sys.dim() as usize;
        let all_spin_half = sys.spins.iter().all(|s| s.isotope.two_i() == 1);

        let sparse = if all_spin_half {
            // Invariant: for an all-spin-½ system, the Hilbert-space dim is
            // exactly 2^N. Worth debug-asserting so the bit-indexed fast path
            // can rely on it without further checks.
            debug_assert_eq!(dim, 1usize << sys.len());
            build_spin_half_jcoupling_sparse(sys.len(), couplings)
        } else {
            let h = build_jcoupling_dense(sys, couplings);
            dense_to_real_sparse(&h)
        };

        Self { dim, sparse }
    }
}

/// Build the J-coupling Hamiltonian as a dense [`Operator`] via kronecker-lifted
/// single-site operators. Cost is O(N_pairs · D³) due to the three D×D matrix
/// products per pair — the reason the spin-½ homonuclear fast path exists.
///
/// Used only as the fallback when at least one spin has I > ½ (currently just
/// 2H homonuclear J-coupling). Validation of `couplings` is the caller's
/// responsibility — see [`JCouplingH::new`].
fn build_jcoupling_dense(sys: &SpinSystem, couplings: &[(usize, usize, f64)]) -> Operator {
    let dim = sys.dim() as usize;
    let two_pi = 2.0 * PI;
    let mut h = Operator::zeros(dim, dim);

    for &(i, j, j_hz) in couplings {
        if j_hz == 0.0 {
            continue;
        }

        let ix_i = ix_at(sys, i);
        let iy_i = iy_at(sys, i);
        let iz_i = iz_at(sys, i);
        let ix_j = ix_at(sys, j);
        let iy_j = iy_at(sys, j);
        let iz_j = iz_at(sys, j);

        let dot = &ix_i * &ix_j + &iy_i * &iy_j + &iz_i * &iz_j;
        let scale = Complex::new(two_pi * j_hz, 0.0);
        h += dot.map(|x| x * scale);
    }
    h
}

/// Direct-sparse construction of H_J for an N-spin-½ homonuclear system.
///
/// Builds the real CSR representation in O(N_pairs · D) time and O(nnz) peak
/// memory without ever materializing the dense D×D operator. This is what
/// unblocks N = 14–17 on a single HPC node: at N=15 the dense path would need
/// ~16 GB just for the cached dense, whereas the sparse has ~N_pairs·D/2
/// off-diagonal entries plus D diagonal entries (~850 k for N=15 with ~100
/// couplings — a few dozen MB).
///
/// # Basis encoding
///
/// We lean on the "first spin varies slowest" convention documented in
/// `operator::lift`: for a system of N spin-½ nuclei, spin `i`'s m-quantum
/// number is encoded in bit `(N - 1 - i)` of the state index. Bit 0 → m = +½
/// (α), bit 1 → m = −½ (β).
///
/// # Structure of H_J per pair
///
/// For a single pair `(i, j, J)`:
///
/// - **Diagonal**: H[k,k] receives `2πJ · m_i(k) · m_j(k)` = ±πJ/2.
/// - **Flip-flop**: at state k where m_i ≠ m_j, the matrix element
///   H[k', k] at k' = k with bits i,j flipped gets `2πJ · ½ = πJ`. (The
///   spin-½ ladder coefficients √(I(I+1) − m(m±1)) are all equal to 1, so
///   the ½ in front of the ladder-form `½(Î⁺ᵢÎ⁻ⱼ + Î⁻ᵢÎ⁺ⱼ)` survives
///   unadorned.)
///
/// # COO duplicate handling
///
/// Multiple couplings may contribute to the same H[k,k]; `CsrMatrix::from(&coo)`
/// sums duplicates. We accumulate the diagonal in a dense `DVector<f64>` first
/// and flush to COO once at the end to keep total COO entries bounded by
/// `N_pairs · D / 2 + D` rather than `N_pairs · D`.
///
/// # Callers
///
/// Validation is assumed done upstream in [`JCouplingH::new`]; this function
/// trusts its inputs.
fn build_spin_half_jcoupling_sparse(n_spins: usize, couplings: &[(usize, usize, f64)]) -> SparseH {
    let dim = 1usize << n_spins;
    let two_pi = 2.0 * PI;
    let mut coo = CooMatrix::<f64>::new(dim, dim);
    let mut diag_accum = DVector::<f64>::zeros(dim);

    for &(i, j, j_hz) in couplings {
        if j_hz == 0.0 {
            continue;
        }

        let scale = two_pi * j_hz;
        let half_scale = 0.5 * scale;
        let shift_i = n_spins - 1 - i;
        let shift_j = n_spins - 1 - j;
        let mask_i = 1usize << shift_i;
        let mask_j = 1usize << shift_j;

        for k in 0..dim {
            let bit_i = (k >> shift_i) & 1;
            let bit_j = (k >> shift_j) & 1;
            let m_i = if bit_i == 0 { 0.5 } else { -0.5 };
            let m_j = if bit_j == 0 { 0.5 } else { -0.5 };

            diag_accum[k] += scale * m_i * m_j;

            if bit_i != bit_j {
                let k_prime = k ^ mask_i ^ mask_j;
                coo.push(k_prime, k, half_scale);
            }
        }
    }

    // Flush the accumulated diagonal. Skip exact zeros so the CSR's `nnz()`
    // reflects true structural sparsity even when e.g. a zero-J pair leaves
    // the diagonal unchanged.
    for (k, &v) in diag_accum.iter().enumerate() {
        if v != 0.0 {
            coo.push(k, k, v);
        }
    }

    CsrMatrix::from(&coo)
}

impl Hamiltonian for JCouplingH {
    fn dim(&self) -> usize {
        self.dim
    }

    fn as_dense(&self) -> Operator {
        // Materialize lazily from the sparse representation. O(nnz) time,
        // O(D²) memory only in the caller's hands — JCouplingH itself never
        // retains a dense. For 10-spin problems (D=1024) the 16 MB allocation
        // is cheap; for N ≥ 15 the caller shouldn't be asking for a dense at
        // all and should use `try_as_sparse` instead.
        let mut m = Operator::zeros(self.dim, self.dim);
        for (i, j, v) in self.sparse.triplet_iter() {
            m[(i, j)] = Complex::new(*v, 0.0);
        }
        m
    }

    // try_as_diagonal: default `None`. JCouplingH is intrinsically non-diagonal
    // whenever any J is nonzero (flip-flop terms). Even in the degenerate
    // "all zero" case we return None — callers that want a fast path for
    // an empty coupling list should just not construct a JCouplingH.

    fn try_as_sparse(&self) -> Option<&SparseH> {
        Some(&self.sparse)
    }
}

/// Additive combinator: `H = H₁ + H₂ + … + Hₙ` over any sequence of
/// [`Hamiltonian`] terms.
///
/// This is how you assemble a full NMR Hamiltonian out of the individual
/// physics contributions — `H = H_Zeeman + H_J + H_dipolar + …` — without
/// any single type needing to know about all of them. New physics terms
/// slot in as new [`Hamiltonian`] implementors and become composable for
/// free.
///
/// # Diagonal fast path
///
/// `try_as_diagonal` returns `Some(&diagonal)` iff every child term
/// advertises diagonal storage. In that case the summed diagonal is
/// precomputed at construction and cached, so the downstream
/// [`DiagonalPropagator`](crate::propagator::DiagonalPropagator) sees the
/// same O(D)-storage representation it would get from a single `ZeemanH`.
/// If any child is non-diagonal (e.g. a [`JCouplingH`]), the combined
/// sum routes through [`MatrixPropagator`](crate::propagator::MatrixPropagator).
///
/// # Panics
///
/// The constructor panics if the term list is empty, or if child dimensions
/// disagree — summing operators on different Hilbert spaces is nonsense.
pub struct SumH {
    terms: Vec<Box<dyn Hamiltonian>>,
    dim: usize,
    /// If every term is diagonal, we precompute the summed diagonal once at
    /// construction. `None` if any term is non-diagonal.
    cached_diagonal: Option<DVector<f64>>,
    /// If every term advertises a sparse view, we precompute the summed sparse
    /// matrix once at construction. `None` if any term's `try_as_sparse` is
    /// `None`. This mirrors `cached_diagonal` and preserves the sparse fast
    /// path through trait composition: Zeeman + JCoupling is still sparse.
    cached_sparse: Option<SparseH>,
}

impl SumH {
    /// Build a sum Hamiltonian from a non-empty list of term boxes.
    pub fn new(terms: Vec<Box<dyn Hamiltonian>>) -> Self {
        assert!(
            !terms.is_empty(),
            "SumH requires at least one term; construct the individual term directly \
             if you only have one"
        );
        let dim = terms[0].dim();
        for (idx, t) in terms.iter().enumerate().skip(1) {
            assert_eq!(
                t.dim(),
                dim,
                "SumH: term {idx} has dim {} but term 0 has dim {dim}",
                t.dim(),
            );
        }

        // Precompute the summed diagonal if every child is diagonal. This
        // preserves the O(D)-storage DiagonalPropagator fast path through
        // trait composition: Zeeman + future diagonal terms still runs fast.
        let all_diagonal = terms.iter().all(|t| t.try_as_diagonal().is_some());
        let cached_diagonal = if all_diagonal {
            let mut d = DVector::<f64>::zeros(dim);
            for t in &terms {
                // unwrap is safe by the all_diagonal check above.
                d += t.try_as_diagonal().unwrap();
            }
            Some(d)
        } else {
            None
        };

        // Precompute the summed sparse if every child is sparse. `+` on CSR
        // matrices in nalgebra-sparse builds a new CSR in structurally-sorted
        // order, so the final `cached_sparse` has canonical CSR layout ready
        // for downstream spmv consumers (Krylov propagator in M4a Step 5).
        //
        // Mathematically sparse-aware Hamiltonians are a superset of diagonal
        // ones — every diagonal term is trivially sparse — but implementors
        // must still opt into both. That's deliberate: the caller's
        // `try_as_sparse` return is a signed contract ("I am stored this way
        // and I claim it's valid"), and we don't want to forge one from the
        // diagonal even when we could.
        let all_sparse = terms.iter().all(|t| t.try_as_sparse().is_some());
        let cached_sparse = if all_sparse {
            let mut iter = terms.iter();
            let first = iter.next().unwrap().try_as_sparse().unwrap().clone();
            let acc = iter.fold(first, |acc, t| &acc + t.try_as_sparse().unwrap());
            Some(acc)
        } else {
            None
        };

        Self {
            terms,
            dim,
            cached_diagonal,
            cached_sparse,
        }
    }

    /// Number of terms in the sum.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Whether this sum is empty. Always `false` because [`new`](Self::new)
    /// rejects empty term lists; kept for symmetry with `len` and to silence
    /// the clippy `len_without_is_empty` lint.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }
}

impl std::fmt::Debug for SumH {
    // Hand-rolled Debug because `Box<dyn Hamiltonian>` doesn't carry Debug.
    // We expose the shape of the sum (count + dim + whether it's diagonal)
    // without demanding Debug from every Hamiltonian implementor.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SumH")
            .field("n_terms", &self.terms.len())
            .field("dim", &self.dim)
            .field("diagonal", &self.cached_diagonal.is_some())
            .field("sparse", &self.cached_sparse.is_some())
            .finish()
    }
}

impl Hamiltonian for SumH {
    fn dim(&self) -> usize {
        self.dim
    }

    fn as_dense(&self) -> Operator {
        let mut m = Operator::zeros(self.dim, self.dim);
        for t in &self.terms {
            m += t.as_dense();
        }
        m
    }

    fn try_as_diagonal(&self) -> Option<&DVector<f64>> {
        self.cached_diagonal.as_ref()
    }

    fn try_as_sparse(&self) -> Option<&SparseH> {
        self.cached_sparse.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spin::{Isotope, Spin};

    /// Reconstruct a dense `Operator` (complex) from a real CSR view. This
    /// is the only way to compare a `SparseH` against the dense ground truth
    /// without pulling a full Kronecker ladder into the test.
    fn sparse_to_dense_complex(s: &SparseH) -> Operator {
        let (nrows, ncols) = (s.nrows(), s.ncols());
        let mut m = Operator::zeros(nrows, ncols);
        for (i, j, v) in s.triplet_iter() {
            m[(i, j)] = Complex::new(*v, 0.0);
        }
        m
    }

    fn two_protons(shifts: [f64; 2]) -> SpinSystem {
        SpinSystem::new(
            vec![
                Spin::new(Isotope::H1, shifts[0]),
                Spin::new(Isotope::H1, shifts[1]),
            ],
            14.0954,
        )
    }

    #[test]
    fn zeeman_of_zero_shift_system_is_zero() {
        // Rotating-frame Zeeman of a system with all shifts at 0 must be
        // exactly zero — by construction, Δωᵢ = 0 for each spin.
        let sys = two_protons([0.0, 0.0]);
        let h = ZeemanH::new(&sys);
        assert_eq!(h.dim(), 4);
        let diag = h.try_as_diagonal().expect("ZeemanH is diagonal");
        for &x in diag.iter() {
            assert!(x.abs() < 1e-12);
        }
    }

    #[test]
    fn zeeman_dense_form_is_diagonal() {
        // The dense reconstruction is, by construction, diagonal. Verify.
        let sys = two_protons([1.0, 5.0]);
        let h = ZeemanH::new(&sys);
        let m = h.as_dense();
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert!(
                        m[(i, j)].norm() < 1e-12,
                        "off-diagonal at ({i},{j}) = {:?} should be zero",
                        m[(i, j)],
                    );
                }
            }
        }
    }

    #[test]
    fn zeeman_is_hermitian() {
        // A real-diagonal operator is trivially Hermitian. Verify via the
        // dense reconstruction.
        let sys = two_protons([1.0, 5.0]);
        let h = ZeemanH::new(&sys);
        let m = h.as_dense();
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                let diff = m[(i, j)] - m[(j, i)].conj();
                assert!(diff.norm() < 1e-12, "not Hermitian at ({i},{j})");
            }
        }
    }

    #[test]
    fn zeeman_is_traceless() {
        // ∑ₘ m = 0 for any Iz representation → every Σᵢ Δωᵢ Îz,ᵢ is traceless.
        let sys = two_protons([1.0, 5.0]);
        let h = ZeemanH::new(&sys);
        let tr: f64 = h.diagonal().iter().sum();
        assert!(tr.abs() < 1e-10, "trace = {tr} should be 0");
    }

    #[test]
    fn zeeman_two_proton_diagonal_matches_hand_computation() {
        // Two 1H spins at 1 ppm and 5 ppm at 14.0954 T.
        //
        // Basis (first spin varies slowest):
        //   |↑↑⟩  index 0: H[0,0] = (+½)Δω₁ + (+½)Δω₂
        //   |↑↓⟩  index 1: H[1,1] = (+½)Δω₁ + (−½)Δω₂
        //   |↓↑⟩  index 2: H[2,2] = (−½)Δω₁ + (+½)Δω₂
        //   |↓↓⟩  index 3: H[3,3] = (−½)Δω₁ + (−½)Δω₂
        let b0 = 14.0954;
        let s1 = Spin::new(Isotope::H1, 1.0);
        let s2 = Spin::new(Isotope::H1, 5.0);
        let d1 = s1.shift_angular(b0);
        let d2 = s2.shift_angular(b0);
        let sys = SpinSystem::new(vec![s1, s2], b0);
        let h = ZeemanH::new(&sys);
        let diag = h.diagonal();

        let expected = [
            0.5 * d1 + 0.5 * d2,
            0.5 * d1 - 0.5 * d2,
            -0.5 * d1 + 0.5 * d2,
            -0.5 * d1 - 0.5 * d2,
        ];
        for (i, &e) in expected.iter().enumerate() {
            assert!(
                (diag[i] - e).abs() < 1e-6,
                "diagonal[{i}]: got {}, expected {}",
                diag[i],
                e,
            );
        }
    }

    #[test]
    fn zeeman_heteronuclear_includes_both_isotopes() {
        // 1H at 7 ppm + 13C at 100 ppm at 14.0954 T.
        // Each isotope in its own rotating frame → diagonal contributions:
        //   H[i,i] = Δω_H · m_H + Δω_C · m_C
        // The bigger magnitude of γ_H means Δω_H ≫ Δω_C even though δ_C > δ_H.
        let b0 = 14.0954;
        let h_spin = Spin::new(Isotope::H1, 7.0);
        let c_spin = Spin::new(Isotope::C13, 100.0);
        let d_h = h_spin.shift_angular(b0);
        let d_c = c_spin.shift_angular(b0);
        let sys = SpinSystem::new(vec![h_spin, c_spin], b0);
        let h = ZeemanH::new(&sys);
        let diag = h.diagonal();

        let expected = [
            0.5 * d_h + 0.5 * d_c,
            0.5 * d_h - 0.5 * d_c,
            -0.5 * d_h + 0.5 * d_c,
            -0.5 * d_h - 0.5 * d_c,
        ];
        for (i, &e) in expected.iter().enumerate() {
            assert!((diag[i] - e).abs() < 1e-6);
        }
    }

    #[test]
    fn hamiltonian_trait_exposes_dimension() {
        let sys = SpinSystem::new(
            vec![
                Spin::new(Isotope::H1, 1.0),
                Spin::new(Isotope::H1, 2.0),
                Spin::new(Isotope::H1, 3.0),
            ],
            14.0954,
        );
        let h: Box<dyn Hamiltonian> = Box::new(ZeemanH::new(&sys));
        // Three spin-1/2's → 2³ = 8.
        assert_eq!(h.dim(), 8);
    }

    #[test]
    fn zeeman_try_as_diagonal_matches_as_dense_diagonal() {
        // The diagonal advertised by `try_as_diagonal` must match the
        // diagonal of the dense reconstruction exactly — same data, two
        // views. Guards against drift if either code path is changed.
        let sys = two_protons([1.5, 4.0]);
        let h = ZeemanH::new(&sys);
        let diag = h.try_as_diagonal().unwrap();
        let m = h.as_dense();
        for k in 0..h.dim() {
            assert!((diag[k] - m[(k, k)].re).abs() < 1e-12);
            assert!(m[(k, k)].im.abs() < 1e-12);
        }
    }

    // ========================================================================
    // JCouplingH
    // ========================================================================

    fn approx_eq_op(a: &Operator, b: &Operator, tol: f64) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        (a - b).iter().all(|x| x.norm() < tol)
    }

    #[test]
    fn jcoupling_empty_list_is_zero() {
        // Construct over no pairs → zero matrix. Defines the identity element
        // of the J-coupling family and exercises the degenerate code path.
        let sys = two_protons([1.0, 5.0]);
        let h = JCouplingH::new(&sys, &[]);
        assert_eq!(h.dim(), 4);
        let m = h.as_dense();
        for i in 0..4 {
            for j in 0..4 {
                assert!(m[(i, j)].norm() < 1e-14);
            }
        }
        // Non-diagonal from the trait's point of view — we don't advertise
        // the "all zeros" special case as diagonal because callers that want
        // the diagonal fast path shouldn't be constructing empty JCouplingHs.
        assert!(h.try_as_diagonal().is_none());
    }

    #[test]
    fn jcoupling_zero_j_is_zero() {
        // Explicit zero-valued J should contribute nothing.
        let sys = two_protons([1.0, 5.0]);
        let h = JCouplingH::new(&sys, &[(0, 1, 0.0)]);
        let m = h.as_dense();
        assert!(m.iter().all(|x| x.norm() < 1e-14));
    }

    #[test]
    fn jcoupling_does_not_advertise_diagonal() {
        // The whole point of this type is that it's non-diagonal in the
        // product-Iz basis. The trait default is `None`; we rely on that.
        let sys = two_protons([1.0, 5.0]);
        let h = JCouplingH::new(&sys, &[(0, 1, 7.0)]);
        assert!(h.try_as_diagonal().is_none());
    }

    #[test]
    fn jcoupling_is_hermitian() {
        // Every Hamiltonian must be Hermitian — spot-check on the full dense
        // form of a realistic 3-spin system with multiple couplings.
        let sys = SpinSystem::new(
            vec![
                Spin::new(Isotope::H1, 1.0),
                Spin::new(Isotope::H1, 2.5),
                Spin::new(Isotope::H1, 7.0),
            ],
            14.0954,
        );
        let h = JCouplingH::new(&sys, &[(0, 1, 7.5), (1, 2, 3.1), (0, 2, 0.6)]);
        let m = h.as_dense();
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                let diff = m[(i, j)] - m[(j, i)].conj();
                assert!(diff.norm() < 1e-10, "not Hermitian at ({i},{j})");
            }
        }
    }

    #[test]
    fn jcoupling_is_traceless() {
        // Îx_i·Îx_j, Îy_i·Îy_j, Îz_i·Îz_j are each traceless (products of
        // traceless operators on different sites), so Σ 2π J Îᵢ·Îⱼ is too.
        let sys = SpinSystem::new(
            vec![
                Spin::new(Isotope::H1, 0.0),
                Spin::new(Isotope::H1, 2.0),
                Spin::new(Isotope::H1, 7.5),
            ],
            14.0954,
        );
        let h = JCouplingH::new(&sys, &[(0, 1, 7.0), (1, 2, 12.0)]);
        let m = h.as_dense();
        let tr = (0..m.nrows()).map(|k| m[(k, k)]).sum::<Complex<f64>>();
        assert!(tr.norm() < 1e-10, "trace = {tr:?} should be 0");
    }

    #[test]
    fn jcoupling_ordering_is_symmetric() {
        // (i, j, J) and (j, i, J) must produce identical operators — the
        // physics has no notion of "first" or "second" spin in a pair.
        let sys = two_protons([2.0, 6.0]);
        let h_ij = JCouplingH::new(&sys, &[(0, 1, 7.3)]);
        let h_ji = JCouplingH::new(&sys, &[(1, 0, 7.3)]);
        assert!(approx_eq_op(&h_ij.as_dense(), &h_ji.as_dense(), 1e-12));
    }

    #[test]
    fn jcoupling_duplicate_pairs_sum() {
        // Documented behavior: duplicate (i, j) entries sum. Verify by
        // comparing (5.0, 3.0) to a single (8.0) entry.
        let sys = two_protons([2.0, 6.0]);
        let h_split = JCouplingH::new(&sys, &[(0, 1, 5.0), (0, 1, 3.0)]);
        let h_merged = JCouplingH::new(&sys, &[(0, 1, 8.0)]);
        assert!(approx_eq_op(
            &h_split.as_dense(),
            &h_merged.as_dense(),
            1e-12
        ));
    }

    #[test]
    #[should_panic(expected = "self-coupling")]
    fn jcoupling_self_coupling_panics() {
        let sys = two_protons([1.0, 5.0]);
        let _ = JCouplingH::new(&sys, &[(1, 1, 5.0)]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn jcoupling_out_of_range_panics() {
        let sys = two_protons([1.0, 5.0]);
        let _ = JCouplingH::new(&sys, &[(0, 2, 5.0)]);
    }

    #[test]
    #[should_panic(expected = "heteronuclear")]
    fn jcoupling_heteronuclear_panics() {
        // Until we implement the secular approximation for mixed isotopes,
        // strong-coupling Îᵢ·Îⱼ across different γs is unphysical here.
        let sys = SpinSystem::new(
            vec![Spin::new(Isotope::H1, 7.0), Spin::new(Isotope::C13, 100.0)],
            14.0954,
        );
        let _ = JCouplingH::new(&sys, &[(0, 1, 120.0)]);
    }

    #[test]
    fn jcoupling_ladder_form_matches_xx_yy_form() {
        // The identity Îx_i·Îx_j + Îy_i·Îy_j = ½(Î⁺ᵢÎ⁻ⱼ + Î⁻ᵢÎ⁺ⱼ) is the
        // algebraic heart of the flip-flop term. Our constructor builds the
        // xx+yy form; this test verifies against the ladder form on a
        // two-spin system, catching any sign error or Kronecker-ordering bug.
        use crate::operator::{iminus_at, iplus_at};
        let sys = two_protons([1.0, 5.0]);
        let j_hz = 7.3;
        let h = JCouplingH::new(&sys, &[(0, 1, j_hz)]);
        let m = h.as_dense();

        // Rebuild via ladder form + zz.
        let ip0 = iplus_at(&sys, 0);
        let im0 = iminus_at(&sys, 0);
        let ip1 = iplus_at(&sys, 1);
        let im1 = iminus_at(&sys, 1);
        let iz0 = iz_at(&sys, 0);
        let iz1 = iz_at(&sys, 1);
        let half = Complex::new(0.5, 0.0);
        let flip_flop = (&ip0 * &im1 + &im0 * &ip1).map(|x| x * half);
        let zz = &iz0 * &iz1;
        let scale = Complex::new(2.0 * PI * j_hz, 0.0);
        let expected = (flip_flop + zz).map(|x| x * scale);

        assert!(approx_eq_op(&m, &expected, 1e-10));
    }

    #[test]
    fn jcoupling_two_spin_matrix_matches_analytic_form() {
        // Classic two-spin strong-coupling Hamiltonian in the |αα⟩, |αβ⟩,
        // |βα⟩, |ββ⟩ basis (first spin varies slowest):
        //
        //   H_J / (2π J) =
        //   | +1/4     0     0     0   |
        //   |   0   -1/4   1/2     0   |
        //   |   0    1/2  -1/4     0   |
        //   |   0     0     0   +1/4   |
        //
        // The off-diagonal 1/2 entries in the M=0 block are the flip-flop
        // terms. This is the reason we built MatrixPropagator.
        let sys = two_protons([0.0, 0.0]);
        let j_hz = 1.0; // unit J so the matrix == (2π) × numerical form above
        let m = JCouplingH::new(&sys, &[(0, 1, j_hz)]).as_dense();
        let scale = 2.0 * PI; // 2π J = 2π

        let expected = |r: f64, c: f64| Complex::new(scale * r, scale * c);
        let q = 0.25;

        // Diagonal
        assert!((m[(0, 0)] - expected(q, 0.0)).norm() < 1e-10);
        assert!((m[(1, 1)] - expected(-q, 0.0)).norm() < 1e-10);
        assert!((m[(2, 2)] - expected(-q, 0.0)).norm() < 1e-10);
        assert!((m[(3, 3)] - expected(q, 0.0)).norm() < 1e-10);
        // Flip-flop
        assert!((m[(1, 2)] - expected(0.5, 0.0)).norm() < 1e-10);
        assert!((m[(2, 1)] - expected(0.5, 0.0)).norm() < 1e-10);
        // Everything else is zero
        for (i, j) in [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 3),
            (2, 0),
            (2, 3),
            (3, 0),
            (3, 1),
            (3, 2),
        ] {
            assert!(
                m[(i, j)].norm() < 1e-10,
                "entry ({i},{j}) = {:?} should be zero",
                m[(i, j)]
            );
        }
    }

    #[test]
    fn jcoupling_commutes_with_total_iz() {
        // [Ĥ_J, M_z] = 0: isotropic coupling conserves total M_z. This is
        // equivalent to the observation that the flip-flop term swaps single-
        // spin projections without changing their sum. Practically it means
        // H_J block-diagonalizes into M-sectors — a property we'll exploit
        // later when we add a restricted-basis propagator.
        let sys = SpinSystem::new(
            vec![
                Spin::new(Isotope::H1, 1.0),
                Spin::new(Isotope::H1, 4.0),
                Spin::new(Isotope::H1, 7.0),
            ],
            14.0954,
        );
        let h = JCouplingH::new(&sys, &[(0, 1, 7.0), (1, 2, 5.0), (0, 2, 1.5)]).as_dense();

        let dim = sys.dim() as usize;
        let mut m_z = Operator::zeros(dim, dim);
        for i in 0..sys.len() {
            m_z += iz_at(&sys, i);
        }
        let commutator = &h * &m_z - &m_z * &h;
        let zero = Operator::zeros(dim, dim);
        assert!(
            approx_eq_op(&commutator, &zero, 1e-8),
            "[H_J, M_z] should vanish"
        );
    }

    // ========================================================================
    // SumH
    // ========================================================================

    #[test]
    fn sumh_single_term_equals_that_term() {
        let sys = two_protons([1.0, 5.0]);
        let h_zeeman = ZeemanH::new(&sys);
        let expected = h_zeeman.as_dense();
        let sum = SumH::new(vec![Box::new(h_zeeman)]);
        assert_eq!(sum.dim(), 4);
        assert!(approx_eq_op(&sum.as_dense(), &expected, 1e-12));
    }

    #[test]
    fn sumh_zeeman_plus_zeeman_preserves_diagonal_fast_path() {
        // Two diagonal children → combined sum is diagonal → try_as_diagonal
        // returns Some, and the returned diagonal is the elementwise sum.
        let sys = two_protons([1.0, 5.0]);
        let a = ZeemanH::new(&sys);
        let b = ZeemanH::new(&sys); // same thing again — doubles the diagonal
        let expected: DVector<f64> = a.diagonal() + b.diagonal();
        let sum = SumH::new(vec![Box::new(a), Box::new(b)]);
        let diag = sum
            .try_as_diagonal()
            .expect("sum of diagonal Hamiltonians must be diagonal");
        for k in 0..sum.dim() {
            assert!((diag[k] - expected[k]).abs() < 1e-12);
        }
    }

    #[test]
    fn sumh_zeeman_plus_jcoupling_is_dense() {
        // Mixed diagonal + non-diagonal children → not diagonal overall.
        // This is the typical NMR H = H_Z + H_J construction.
        let sys = two_protons([1.0, 5.0]);
        let hz = ZeemanH::new(&sys);
        let hj = JCouplingH::new(&sys, &[(0, 1, 7.0)]);
        let expected = hz.as_dense() + hj.as_dense();

        let sum = SumH::new(vec![Box::new(hz), Box::new(hj)]);
        assert!(sum.try_as_diagonal().is_none());
        assert!(approx_eq_op(&sum.as_dense(), &expected, 1e-12));
    }

    #[test]
    fn sumh_associativity_on_three_terms() {
        // (A + B) + C  ==  A + (B + C) as dense matrices. Mild sanity check
        // that SumH doesn't accumulate order-dependent roundoff.
        let sys = two_protons([1.0, 5.0]);
        let build = || {
            (
                ZeemanH::new(&sys),
                JCouplingH::new(&sys, &[(0, 1, 3.0)]),
                JCouplingH::new(&sys, &[(0, 1, 4.0)]),
            )
        };

        let (a1, b1, c1) = build();
        let left = SumH::new(vec![
            Box::new(SumH::new(vec![Box::new(a1), Box::new(b1)])),
            Box::new(c1),
        ])
        .as_dense();

        let (a2, b2, c2) = build();
        let right = SumH::new(vec![
            Box::new(a2),
            Box::new(SumH::new(vec![Box::new(b2), Box::new(c2)])),
        ])
        .as_dense();

        assert!(approx_eq_op(&left, &right, 1e-10));
    }

    #[test]
    #[should_panic(expected = "at least one term")]
    fn sumh_empty_panics() {
        let _ = SumH::new(Vec::<Box<dyn Hamiltonian>>::new());
    }

    #[test]
    #[should_panic(expected = "dim")]
    fn sumh_dim_mismatch_panics() {
        // 2 spins vs 3 spins → dim 4 vs dim 8. Refuse to compose.
        let sys2 = two_protons([1.0, 5.0]);
        let sys3 = SpinSystem::new(
            vec![
                Spin::new(Isotope::H1, 1.0),
                Spin::new(Isotope::H1, 4.0),
                Spin::new(Isotope::H1, 7.0),
            ],
            14.0954,
        );
        let a = ZeemanH::new(&sys2);
        let b = ZeemanH::new(&sys3);
        let _ = SumH::new(vec![Box::new(a), Box::new(b)]);
    }

    // ========================================================================
    // AB system: analytical validation
    // ========================================================================

    #[test]
    fn two_spin_strong_coupling_eigenvalues_match_analytic_ab() {
        // Classic AB system: two protons with chemical-shift difference Δν
        // and coupling J. The eigenvalues of H_Z + H_J (in Hz units, after
        // dividing by 2π) are analytically:
        //
        //   E_αα = +½(ν_A + ν_B) + J/4
        //   E_inner± = −J/4 ± ½√(Δν² + J²)
        //   E_ββ = −½(ν_A + ν_B) + J/4
        //
        // We construct the rotating-frame Hamiltonian (so ν_A, ν_B are just
        // the chemical-shift offsets in Hz), diagonalize via symmetric
        // eigendecomposition, and check the four eigenvalues against the
        // closed form. Because the eigendecomposition gives a scrambled
        // order, we sort both sets before comparing.
        use nalgebra::DMatrix;

        let b0 = 14.0954;
        let gamma_h = Isotope::H1.gamma();
        // Chemical shifts chosen so |Δν| ~ 2.5 × J — deeply in the strong-
        // coupling regime, AKA roofing regime. Test is a harder challenge
        // for the numerics the further from first-order we push.
        let delta_ppm = 0.8;
        let sum_ppm = 4.0;
        let shift_a = sum_ppm * 0.5 + delta_ppm * 0.5;
        let shift_b = sum_ppm * 0.5 - delta_ppm * 0.5;
        let j_hz: f64 = 7.5;

        // Expected frequencies (Hz) — convert shifts to Hz via ν = |γ| B / 2π × δ_ppm × 1e-6.
        let nu_a = gamma_h.abs() * b0 / (2.0 * PI) * shift_a * 1e-6;
        let nu_b = gamma_h.abs() * b0 / (2.0 * PI) * shift_b * 1e-6;
        let c_hz = 0.5 * ((nu_a - nu_b).powi(2) + j_hz.powi(2)).sqrt();

        // Our shift_angular sign convention: Δω = −γ B δ. For γ > 0 (1H)
        // that makes Δω_A *negative* when shift_A > 0. The frequencies we
        // defined above are positive; we need to align signs. ν_A and ν_B
        // as defined by our rotating-frame convention are −nu_a, −nu_b in
        // Hz. But eigenvalues of H depend on (ν_A + ν_B) and |Δν|, both
        // of which flip sign consistently — so we compare magnitudes by
        // taking the signed ν's from the spin system itself.
        let sys = SpinSystem::new(
            vec![
                Spin::new(Isotope::H1, shift_a),
                Spin::new(Isotope::H1, shift_b),
            ],
            b0,
        );
        let true_nu_a = sys.spins[0].shift_angular(b0) / (2.0 * PI);
        let true_nu_b = sys.spins[1].shift_angular(b0) / (2.0 * PI);
        let true_sum = true_nu_a + true_nu_b;
        let true_c = 0.5 * ((true_nu_a - true_nu_b).powi(2) + j_hz.powi(2)).sqrt();
        // Confirm our hand-computed |c_hz| matches the sys-derived version.
        assert!((c_hz - true_c).abs() < 1e-6);

        // Build the full H (rad/s), then convert to Hz by dividing by 2π.
        let h = SumH::new(vec![
            Box::new(ZeemanH::new(&sys)),
            Box::new(JCouplingH::new(&sys, &[(0, 1, j_hz)])),
        ]);
        let m_rad = h.as_dense();
        // Convert Operator (complex) to real DMatrix<f64>. H is Hermitian,
        // and for this particular H it is in fact real (no Îy appears
        // once we expand via ladder form over a homonuclear zz + flip-flop).
        // Verify before casting.
        for i in 0..m_rad.nrows() {
            for j in 0..m_rad.ncols() {
                assert!(
                    m_rad[(i, j)].im.abs() < 1e-10,
                    "H should be real at ({i},{j}): {:?}",
                    m_rad[(i, j)],
                );
            }
        }
        let m_hz: DMatrix<f64> = DMatrix::from_fn(m_rad.nrows(), m_rad.ncols(), |i, j| {
            m_rad[(i, j)].re / (2.0 * PI)
        });

        let eig = m_hz.symmetric_eigen();
        let mut got: Vec<f64> = eig.eigenvalues.iter().copied().collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut expected = [
            0.5 * true_sum + j_hz / 4.0,
            -0.5 * true_sum + j_hz / 4.0,
            -j_hz / 4.0 + true_c,
            -j_hz / 4.0 - true_c,
        ];
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-6,
                "eigenvalue mismatch: got {g}, expected {e}",
            );
        }
    }

    // ========================================================================
    // Sparse (CSR) fast path
    // ========================================================================

    #[test]
    fn zeeman_sparse_matches_dense() {
        // The cached sparse view must agree with the dense reconstruction
        // element-by-element. Covers the diagonal_to_sparse helper and the
        // ZeemanH::try_as_sparse trait impl.
        let sys = two_protons([1.0, 5.0]);
        let h = ZeemanH::new(&sys);
        let dense = h.as_dense();
        let sparse = h.try_as_sparse().expect("ZeemanH advertises sparse");
        let sparse_dense = sparse_to_dense_complex(sparse);
        assert!(approx_eq_op(&dense, &sparse_dense, 1e-12));
        // Zero entries on the diagonal (none expected here with shifts 1 and
        // 5 ppm, but the code path) are dropped from the CSR. nnz ≤ dim.
        assert!(sparse.nnz() <= h.dim());
    }

    #[test]
    fn zeeman_sparse_drops_zero_diagonal_entries() {
        // A two-proton system with both shifts at zero has an exactly-zero
        // diagonal. The sparse view must have zero nnz — not four zero
        // entries masquerading as structure. Keeping this clean matters
        // because SumH sums CSR matrices structurally, and spurious zeros
        // inflate the combined sparse pattern.
        let sys = two_protons([0.0, 0.0]);
        let h = ZeemanH::new(&sys);
        let sparse = h.try_as_sparse().unwrap();
        assert_eq!(sparse.nnz(), 0);
    }

    #[test]
    fn jcoupling_sparse_matches_dense() {
        // JCouplingH's sparse view is extracted from the dense matrix via
        // dense_to_real_sparse, which asserts imaginary residuals are below
        // 1e-12. This test exercises that path on a 3-spin system with
        // multiple flip-flop-active couplings — the most stressful realistic
        // case at this dim before benchmarks enter the picture.
        let sys = SpinSystem::new(
            vec![
                Spin::new(Isotope::H1, 1.0),
                Spin::new(Isotope::H1, 2.5),
                Spin::new(Isotope::H1, 7.0),
            ],
            14.0954,
        );
        let h = JCouplingH::new(&sys, &[(0, 1, 7.5), (1, 2, 3.1), (0, 2, 0.6)]);
        let dense = h.as_dense();
        let sparse = h.try_as_sparse().expect("JCouplingH advertises sparse");
        let sparse_dense = sparse_to_dense_complex(sparse);
        assert!(approx_eq_op(&dense, &sparse_dense, 1e-10));
    }

    #[test]
    fn jcoupling_sparse_is_strictly_sparser_than_dense_at_modest_n() {
        // For three spins (D=8) with three pairwise couplings, the H_J dense
        // matrix has a lot of structural zeros: |αβγ⟩ and |γβα⟩ do not couple
        // if their M_z differ. This sanity-checks that we're storing genuinely
        // fewer entries than D² — the whole *point* of the sparse fast path.
        let sys = SpinSystem::new(
            vec![
                Spin::new(Isotope::H1, 1.0),
                Spin::new(Isotope::H1, 2.0),
                Spin::new(Isotope::H1, 3.0),
            ],
            14.0954,
        );
        let h = JCouplingH::new(&sys, &[(0, 1, 7.0), (1, 2, 5.0), (0, 2, 1.0)]);
        let sparse = h.try_as_sparse().unwrap();
        let d = h.dim();
        assert!(
            sparse.nnz() < d * d,
            "sparse nnz {} should be strictly less than dense D² = {}",
            sparse.nnz(),
            d * d,
        );
    }

    #[test]
    fn sumh_sparse_composes_when_all_children_are_sparse() {
        // Zeeman + JCoupling: both advertise sparse, so the SumH caches a
        // summed CSR. The summed sparse must reconstruct to the same dense
        // as SumH::as_dense. This is the production path: every propagator
        // in M4a builds an H by summing Zeeman + JCoupling and asks for the
        // sparse view.
        let sys = two_protons([1.0, 5.0]);
        let hz = ZeemanH::new(&sys);
        let hj = JCouplingH::new(&sys, &[(0, 1, 7.0)]);
        let sum = SumH::new(vec![Box::new(hz), Box::new(hj)]);
        let sparse = sum
            .try_as_sparse()
            .expect("both children are sparse; sum must be too");
        let sparse_dense = sparse_to_dense_complex(sparse);
        assert!(approx_eq_op(&sum.as_dense(), &sparse_dense, 1e-10));
    }

    #[test]
    fn jcoupling_h2_homonuclear_falls_back_and_stays_hermitian() {
        // Two deuterium (I=1) nuclei with a J-coupling. This is the only
        // homonuclear isotope we currently support where the spin-½ fast path
        // *doesn't* apply — the mixed-radix fallback kicks in via
        // build_jcoupling_dense. The fallback must still produce a Hermitian,
        // traceless, structurally-sparse H. D = 3² = 9, so the matrix is
        // small enough to enumerate.
        let sys = SpinSystem::new(
            vec![Spin::new(Isotope::H2, 0.0), Spin::new(Isotope::H2, 0.0)],
            14.0954,
        );
        let h = JCouplingH::new(&sys, &[(0, 1, 5.0)]);
        assert_eq!(h.dim(), 9);

        // Hermitian.
        let m = h.as_dense();
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                let diff = m[(i, j)] - m[(j, i)].conj();
                assert!(diff.norm() < 1e-10, "not Hermitian at ({i},{j})");
            }
        }

        // Traceless: Îᵢ·Îⱼ is traceless for any I, so 2πJ times it is too.
        let tr: Complex<f64> = (0..m.nrows()).map(|k| m[(k, k)]).sum();
        assert!(tr.norm() < 1e-10, "trace = {tr:?} should be 0");

        // Sparse view is advertised and reconstructs to the same dense.
        let sparse = h.try_as_sparse().expect("fallback still advertises sparse");
        let rebuilt = sparse_to_dense_complex(sparse);
        assert!(approx_eq_op(&m, &rebuilt, 1e-10));

        // And the CSR is structurally sparser than dense D² = 81.
        assert!(sparse.nnz() < 81);
    }

    #[test]
    fn jcoupling_spin_half_fast_path_matches_hand_computed_four_spin() {
        // Four-spin-½ system with two couplings (0,1) and (2,3), no coupling
        // between the halves. H_J factors into two independent two-spin blocks
        // that are tensor-added. This exercises the direct-sparse path on a
        // non-trivial bit pattern where the naive "XOR two bits" logic must be
        // correct for asymmetric pair positions.
        //
        // We verify by comparing to a hand-built dense reconstruction using
        // kronecker products, which the direct-sparse code never touches.
        let sys = SpinSystem::new(
            vec![
                Spin::new(Isotope::H1, 0.0),
                Spin::new(Isotope::H1, 0.0),
                Spin::new(Isotope::H1, 0.0),
                Spin::new(Isotope::H1, 0.0),
            ],
            14.0954,
        );
        let j01 = 7.0;
        let j23 = 3.5;
        let h = JCouplingH::new(&sys, &[(0, 1, j01), (2, 3, j23)]);
        assert_eq!(h.dim(), 16);

        // Hand build via operator::*_at + kronecker — same math, totally
        // different code path (the whole operator crate hasn't been touched
        // since M1). If the fast path has an encoding bug, this catches it.
        let make_dot = |i: usize, j: usize, j_hz: f64| -> Operator {
            let ix_i = ix_at(&sys, i);
            let iy_i = iy_at(&sys, i);
            let iz_i = iz_at(&sys, i);
            let ix_j = ix_at(&sys, j);
            let iy_j = iy_at(&sys, j);
            let iz_j = iz_at(&sys, j);
            let dot = &ix_i * &ix_j + &iy_i * &iy_j + &iz_i * &iz_j;
            let scale = Complex::new(2.0 * PI * j_hz, 0.0);
            dot.map(|x| x * scale)
        };
        let expected = make_dot(0, 1, j01) + make_dot(2, 3, j23);
        assert!(approx_eq_op(&h.as_dense(), &expected, 1e-10));
    }

    #[test]
    fn sumh_sparse_returns_none_if_any_child_is_not_sparse() {
        // If a third-party Hamiltonian implementor ever exposes only
        // `as_dense`, SumH must honor that and decline to forge a sparse
        // view. We use a local stub as the non-sparse child to prove the
        // negative case.
        struct DenseOnly(Operator);
        impl Hamiltonian for DenseOnly {
            fn dim(&self) -> usize {
                self.0.nrows()
            }
            fn as_dense(&self) -> Operator {
                self.0.clone()
            }
            // Intentionally leaves try_as_diagonal and try_as_sparse as None.
        }

        let sys = two_protons([1.0, 5.0]);
        let hz = ZeemanH::new(&sys);
        let opaque = DenseOnly(Operator::zeros(sys.dim() as usize, sys.dim() as usize));

        let sum = SumH::new(vec![Box::new(hz), Box::new(opaque)]);
        assert!(
            sum.try_as_sparse().is_none(),
            "SumH must not forge a sparse view when a child declines to advertise one"
        );
        // Adding a zero-filled opaque term must not perturb the dense form.
        let sys2 = two_protons([1.0, 5.0]);
        let expected = ZeemanH::new(&sys2).as_dense();
        assert!(approx_eq_op(&sum.as_dense(), &expected, 1e-12));
    }
}
