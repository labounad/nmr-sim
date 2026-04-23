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
//! The trait exposes two views of the underlying operator:
//!
//! - [`as_dense`](Hamiltonian::as_dense): a full `D×D` dense matrix.
//!   Always available; the universal fallback.
//! - [`try_as_diagonal`](Hamiltonian::try_as_diagonal): the diagonal of H as
//!   a real `DVector<f64>`, if the Hamiltonian is stored diagonally in the
//!   product-Iz basis. Returns `None` by default.
//!
//! This split lets propagators specialize without paying for diagonality
//! detection on every construction. A [`DiagonalPropagator`](crate::propagator::DiagonalPropagator)
//! asks `try_as_diagonal` and either gets the data it needs in O(D) memory
//! or refuses the job up-front. A general [`MatrixPropagator`](crate::propagator::MatrixPropagator)
//! always calls `as_dense` and pays the O(D²) cost.
//!
//! # Current implementations
//!
//! - [`ZeemanH`]: isotropic chemical-shift (rotating-frame Zeeman) Hamiltonian.
//!   Diagonal in the product-Iz basis; stored as `DVector<f64>`.
//! - [`JCouplingH`]: isotropic scalar coupling Σᵢ<ⱼ 2π Jᵢⱼ Îᵢ·Îⱼ. Non-diagonal
//!   in the product-Iz basis (the flip-flop terms ½(Î⁺ᵢÎ⁻ⱼ + Î⁻ᵢÎ⁺ⱼ) are
//!   off-diagonal), so this type implements `as_dense` only — callers are
//!   automatically routed through [`MatrixPropagator`](crate::propagator::MatrixPropagator).
//! - [`SumH`]: additive combinator over any sequence of [`Hamiltonian`] terms.
//!   The standard way to write `H = H_Z + H_J + …`. Advertises `try_as_diagonal`
//!   iff every child does, so `ZeemanH + anotherDiagonalTerm` still routes
//!   through the O(D) diagonal fast path.
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
use num_complex::Complex;
use std::f64::consts::PI;

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
        Self { diagonal }
    }

    /// Borrow the real diagonal directly. Equivalent to
    /// `self.try_as_diagonal().unwrap()` but statically infallible.
    pub fn diagonal(&self) -> &DVector<f64> {
        &self.diagonal
    }
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
/// The dense matrix is materialized once in [`new`](Self::new) and cached —
/// there is no more-structured representation to defer to (for now). A future
/// sparse or matrix-free variant can coexist as a sibling type without
/// touching this one.
#[derive(Debug, Clone)]
pub struct JCouplingH {
    /// Cached dense matrix in rad/s.
    dense: Operator,
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
        let dim = sys.dim() as usize;
        let mut h = Operator::zeros(dim, dim);
        let two_pi = 2.0 * PI;

        for &(i, j, j_hz) in couplings {
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

            if j_hz == 0.0 {
                continue;
            }

            // Build Îᵢ·Îⱼ = Îx,ᵢÎx,ⱼ + Îy,ᵢÎy,ⱼ + Îz,ᵢÎz,ⱼ in the full Hilbert
            // space. Each single-site operator is lifted via kronecker in
            // `operator::*_at`. The cost per pair is O(D² · N) for the lifts
            // plus O(D³) for the three matrix products; for small systems
            // (≤ ~12 spin-1/2) this is fine. Future optimization path: build
            // the sparse representation directly and skip the dense matmuls.
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

        Self { dense: h }
    }

    /// Borrow the cached dense matrix without cloning. Cheaper than
    /// [`as_dense`](Self::as_dense) when you only need a read-only view.
    pub fn matrix(&self) -> &Operator {
        &self.dense
    }
}

impl Hamiltonian for JCouplingH {
    fn dim(&self) -> usize {
        self.dense.nrows()
    }

    fn as_dense(&self) -> Operator {
        self.dense.clone()
    }

    // try_as_diagonal: default `None`. JCouplingH is intrinsically non-diagonal
    // whenever any J is nonzero (flip-flop terms). Even in the degenerate
    // "all zero" case we return None — callers that want a fast path for
    // an empty coupling list should just not construct a JCouplingH.
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

        Self {
            terms,
            dim,
            cached_diagonal,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spin::{Isotope, Spin};

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
}
