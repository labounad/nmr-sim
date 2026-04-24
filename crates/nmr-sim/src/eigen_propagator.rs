//! Eigendecomposition propagator for time-independent Hamiltonians.
//!
//! When H is time-independent, diagonalizing once up front and then
//! propagating in its own eigenbasis reduces per-step cost from O(D³)
//! (the [`MatrixPropagator`](crate::propagator::MatrixPropagator) triple
//! product) to O(D²). Setup cost is one O(D³) diagonalization plus a
//! change-of-basis transform on ρ and Ô.
//!
//! # When this wins
//!
//! - Large D where [`MatrixPropagator`](crate::propagator::MatrixPropagator)'s
//!   per-step O(D³) starts to dominate (≳ a few hundred — roughly 8 spin-1/2's
//!   or more).
//! - Long FIDs (high `n_points`), where the up-front O(D³) diagonalization
//!   is amortized across many cheap steps.
//! - Time-*independent* H only. RF pulses and other time-dependent terms will
//!   need a different backend (piecewise-constant propagator, or Magnus /
//!   Krylov).
//!
//! # The math
//!
//! For Hermitian H, eigendecomposition gives
//!
//! ```text
//! H = V · diag(E) · V†
//! ```
//!
//! so the unitary is just
//!
//! ```text
//! U = exp(−iHΔt) = V · diag(exp(−iEₖΔt)) · V†.
//! ```
//!
//! Let `ρ̃ = V† ρ V` be ρ expressed in the eigenbasis. Then
//!
//! ```text
//! U ρ U† = V · (D ρ̃ D†) · V†,        D = diag(exp(−iEₖΔt))
//! (D ρ̃ D†)[i, j] = exp(−i(Eᵢ − Eⱼ)Δt) · ρ̃[i, j]
//! ```
//!
//! i.e. the triple product inside the parentheses is just *elementwise*
//! multiplication.
//!
//! # Slow vs. fast paths
//!
//! The [`Propagator`] trait's [`apply`](Propagator::apply) method would
//! require transforming in and out of the eigenbasis on every call — two
//! O(D³) matmuls per step, which is actually *slower* than
//! [`MatrixPropagator`]. So `apply` exists only as a correctness reference
//! and a trait-satisfying fallback.
//!
//! For real use, [`EigenPropagator::compute_fid`] bypasses the trait and
//! stays in the eigenbasis for every step. It transforms ρ and Ô once, then
//! rewrites the FID sample as
//!
//! ```text
//! s(t) = Tr[Ô · ρ(t)] = Σᵢⱼ A[i, j] · aₜ[i] · aₜ*[j]
//! ```
//!
//! where `A[i, j] = Ỗ[j, i] · ρ̃(0)[i, j]` is a fixed amplitude matrix and
//! `aₜ[i] = exp(−iEᵢt)` is a single complex vector updated multiplicatively
//! each step (`aₖ₊₁[i] = aₖ[i] · exp(−iEᵢΔt)`). Per-step cost is one
//! matrix-vector product plus one dot product — O(D²), no matmuls in the
//! hot loop.
//!
//! # Real-symmetric H optimization
//!
//! Every Hamiltonian currently in this crate ([`ZeemanH`](crate::ZeemanH),
//! [`JCouplingH`](crate::JCouplingH), and [`SumH`](crate::SumH) of them)
//! comes out purely real in the product-Iz basis: the Îy-Îy cross terms in
//! Îᵢ·Îⱼ exactly cancel the imaginary parts of Îx·Îx + Îy·Îy + Îz·Îz.
//! The constructor asserts this (`|Im H[i, j]| < REAL_TOL`) and then runs
//! nalgebra's real `SymmetricEigen`, which is meaningfully faster and more
//! numerically robust than a complex-Hermitian eigensolver. If a future
//! Hamiltonian term genuinely has complex matrix elements (e.g. a rotating-
//! frame off-resonance RF pulse), the assertion will catch the mistake and
//! the solution will be to add a complex-Hermitian constructor variant
//! (or route to [`MatrixPropagator`]).
//!
//! # Parallel LAPACK backend
//!
//! By default, the eigendecomposition uses nalgebra's pure-Rust
//! `symmetric_eigen`, which is portable and single-threaded. Enabling the
//! `parallel-lapack` cargo feature (via one of `lapack-accelerate`,
//! `lapack-openblas`, `lapack-intel-mkl`, or `lapack-netlib`) swaps in
//! `nalgebra-lapack`'s threaded system-LAPACK implementation. The switch is
//! contained in the `symmetric_eigen_real` helper below; nothing else in the
//! propagator changes. At N = 15 (D = 32768) the LAPACK path is the
//! difference between a one-minute setup and a "just go get coffee" setup.

use crate::hamiltonian::Hamiltonian;
use crate::operator::Operator;
use crate::propagator::Propagator;
use crate::state::DensityMatrix;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;

/// Human-readable label for whichever eigendecomposition backend was compiled
/// in. Intended for benchmark / example output so "this run took 0.3 s" is
/// attributable to a specific backend. Evaluated at compile time — `cfg!`
/// inside a `const` context expands to boolean literals and folds away.
#[cfg(feature = "parallel-lapack")]
pub const EIGEN_BACKEND: &str = if cfg!(feature = "lapack-accelerate") {
    "nalgebra-lapack (Accelerate)"
} else if cfg!(feature = "lapack-openblas") {
    "nalgebra-lapack (OpenBLAS)"
} else if cfg!(feature = "lapack-intel-mkl") {
    "nalgebra-lapack (Intel MKL)"
} else if cfg!(feature = "lapack-netlib") {
    "nalgebra-lapack (Netlib)"
} else {
    // parallel-lapack on but no specific backend selected — would fail at
    // link time, but label honestly if it ever gets this far.
    "nalgebra-lapack (unspecified backend)"
};

/// Human-readable label for the eigendecomposition backend. See the
/// `parallel-lapack` cfg-branch above for the feature-enabled variants.
#[cfg(not(feature = "parallel-lapack"))]
pub const EIGEN_BACKEND: &str = "nalgebra symmetric_eigen (pure-Rust)";

/// Diagonalize a real-symmetric matrix, returning `(eigenvalues, eigenvectors)`
/// where the eigenvectors are columns of the returned matrix.
///
/// Dispatches to nalgebra-lapack (threaded system BLAS/LAPACK) when the
/// `parallel-lapack` feature is enabled, and falls back to nalgebra's
/// pure-Rust `symmetric_eigen` otherwise. The two backends produce the same
/// eigenvalues (up to roundoff) and the same eigenvectors up to per-column
/// sign, which is invariant of the propagator since V and V† enter
/// symmetrically in every downstream expression.
#[cfg(feature = "parallel-lapack")]
fn symmetric_eigen_real(h_sym: DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>) {
    let eig = nalgebra_lapack::SymmetricEigen::new(h_sym);
    (eig.eigenvalues, eig.eigenvectors)
}

#[cfg(not(feature = "parallel-lapack"))]
fn symmetric_eigen_real(h_sym: DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>) {
    let eig = h_sym.symmetric_eigen();
    (eig.eigenvalues, eig.eigenvectors)
}

/// Propagator specialized to a time-independent Hermitian Hamiltonian via
/// one-shot eigendecomposition.
///
/// Builds and caches `V`, `V†`, the eigenvalues `Eₖ`, and the per-step phase
/// factors `exp(−iEₖΔt)`. See the module-level docs for the algorithmic
/// trade-offs.
pub struct EigenPropagator {
    /// `V` (D×D): columns are eigenvectors. Stored as a complex matrix
    /// because every downstream use multiplies it against complex operators
    /// / density matrices, so we pay the widening cost exactly once at
    /// construction rather than on every hot-loop multiply.
    v: Operator,
    /// `V†` cached at construction. Also stored complex for the same reason.
    v_dagger: Operator,
    /// Real eigenvalues of H (rad/s). H is Hermitian → eigenvalues are real.
    eigenvalues: DVector<f64>,
    /// Per-eigenstate single-step phase factor `exp(−iEₖΔt)`.
    phases: DVector<Complex<f64>>,
    /// The time step this propagator advances ρ by per [`apply`] call.
    dt: f64,
}

impl EigenPropagator {
    /// Tolerance on `|Im H[i, j]|` when casting H to a real symmetric matrix.
    ///
    /// All currently-shipped Hamiltonians are real in the product-Iz basis
    /// to roundoff; we pick a tolerance generous enough to let ~10⁻⁹ · ‖H‖
    /// through but tight enough to catch a real bug (like accidentally
    /// introducing an Îy-only term).
    const REAL_TOL: f64 = 1e-8;

    /// Build an `EigenPropagator` from any [`Hamiltonian`].
    ///
    /// # Panics
    ///
    /// - If `h.as_dense()` is not square.
    /// - If any matrix element has `|Im H[i, j]| > REAL_TOL`. In that case
    ///   use [`MatrixPropagator`](crate::propagator::MatrixPropagator)
    ///   instead (or add a complex-Hermitian variant of this backend).
    pub fn new<H: Hamiltonian + ?Sized>(h: &H, dt: f64) -> Self {
        let m = h.as_dense();
        Self::from_matrix(&m, dt)
    }

    /// Build an `EigenPropagator` directly from an already-materialized dense
    /// Hamiltonian matrix. The caller is responsible for ensuring `h` is
    /// Hermitian (and real-symmetric in whatever basis it is expressed in —
    /// see [`REAL_TOL`](Self::REAL_TOL)) in rad/s units.
    #[must_use]
    pub fn from_matrix(h: &Operator, dt: f64) -> Self {
        let n = h.nrows();
        assert_eq!(n, h.ncols(), "Hamiltonian must be square");

        // Confirm H is real (within tolerance) before we cast away the
        // imaginary parts. Any genuinely complex Hermitian term would make
        // the real-symmetric path mathematically wrong.
        let max_imag = h.iter().map(|c| c.im.abs()).fold(0.0_f64, f64::max);
        assert!(
            max_imag < Self::REAL_TOL,
            "EigenPropagator: Hamiltonian has complex entries (max |Im H| = \
             {max_imag:.3e} > tol {tol:.0e}). This backend currently requires \
             a real-symmetric H in the product-Iz basis. For a genuinely \
             complex Hermitian H, use MatrixPropagator.",
            tol = Self::REAL_TOL,
        );

        // Cast to DMatrix<f64>. Symmetrize to absorb any O(ε) asymmetry from
        // roundoff during as_dense(): nalgebra's `symmetric_eigen` assumes
        // *exact* symmetry and only inspects one triangle.
        let h_real = DMatrix::<f64>::from_fn(n, n, |i, j| h[(i, j)].re);
        let h_sym = (&h_real + h_real.transpose()) * 0.5;

        let (eigenvalues, eigenvectors) = symmetric_eigen_real(h_sym);

        // Lift eigenvectors back to complex so downstream ops (which are all
        // complex) don't need per-multiply type promotion.
        let v = Operator::from_fn(n, n, |i, j| Complex::new(eigenvectors[(i, j)], 0.0));
        let v_dagger = v.adjoint();

        let phases = DVector::from_iterator(
            n,
            eigenvalues
                .iter()
                .map(|&e| Complex::from_polar(1.0, -e * dt)),
        );

        Self {
            v,
            v_dagger,
            eigenvalues,
            phases,
            dt,
        }
    }

    /// Borrow the eigenvalues of H (rad/s). Diagnostic / analysis hook.
    pub fn eigenvalues(&self) -> &DVector<f64> {
        &self.eigenvalues
    }

    /// Borrow the eigenvector matrix V (columns = eigenstates of H in the
    /// product-Iz basis). Diagnostic / analysis hook.
    pub fn eigenvectors(&self) -> &Operator {
        &self.v
    }

    /// Compute an FID without ever leaving the eigenbasis — the *fast* path.
    ///
    /// Equivalent to [`crate::fid::compute_fid`] called with this propagator
    /// as the `P: Propagator` argument, but faster by a factor of roughly D:
    /// the trait path incurs two O(D³) basis transforms per step, while this
    /// method transforms ρ and Ô once and then does an O(D²) update per step.
    ///
    /// # Complexity
    ///
    /// - Setup: one O(D²) transform of ρ, one of Ô, plus a D×D amplitude
    ///   matrix.
    /// - Per step: one D×D complex mat-vec + one D-dim dot + one D-dim
    ///   componentwise mul = O(D²).
    /// - Total: O(N·D²) after the O(D³) eigendecomposition done in the
    ///   constructor.
    ///
    /// # Panics
    ///
    /// - If either `rho_initial` or `observable` is not square or has
    ///   dimension ≠ the propagator's dimension.
    pub fn compute_fid(
        &self,
        rho_initial: &DensityMatrix,
        observable: &Operator,
        n_points: usize,
    ) -> Vec<Complex<f64>> {
        let d = self.v.nrows();
        assert_eq!(rho_initial.nrows(), d, "ρ dim must match propagator dim");
        assert_eq!(rho_initial.ncols(), d, "ρ must be square");
        assert_eq!(
            observable.nrows(),
            d,
            "observable dim must match propagator dim",
        );
        assert_eq!(observable.ncols(), d, "observable must be square");

        // Transform ρ(0) and Ô into the eigenbasis once.
        let rho_eig = &self.v_dagger * rho_initial * &self.v;
        let obs_eig = &self.v_dagger * observable * &self.v;

        // A[i, j] = Ỗ[j, i] · ρ̃(0)[i, j] — the amplitude of the {i, j}
        // coherence in the observed FID. Derivation:
        //
        //   s(t) = Tr[Ỗ · ρ̃(t)]
        //        = Σᵢⱼ Ỗ[i, j] · ρ̃(t)[j, i]
        //        = Σᵢⱼ Ỗ[i, j] · ρ̃(0)[j, i] · exp(−i(Eⱼ − Eᵢ)t)
        //
        // Swap dummy indices (i ↔ j) to get
        //
        //   s(t) = Σᵢⱼ Ỗ[j, i] · ρ̃(0)[i, j] · exp(−i(Eᵢ − Eⱼ)t)
        //        = Σᵢⱼ A[i, j] · aₜ[i] · aₜ*[j]
        //
        // with A as defined above and aₜ[i] = exp(−iEᵢt).
        let a_mat = Operator::from_fn(d, d, |i, j| obs_eig[(j, i)] * rho_eig[(i, j)]);

        // Maintain aₖ incrementally: a₀ = 1; aₖ₊₁ = aₖ ⊙ phases.
        //
        // Per-step FID sample in matrix form:
        //   s[k] = aₖᵀ · A · aₖ*
        // computed as (A · aₖ*) first (O(D²) mat-vec), then aₖᵀ · (that)
        // (O(D) dot).
        let mut a_k: DVector<Complex<f64>> = DVector::from_element(d, Complex::new(1.0, 0.0));

        let mut fid = Vec::with_capacity(n_points);
        for _ in 0..n_points {
            let a_conj = a_k.map(|c| c.conj());
            let tmp = &a_mat * &a_conj;
            let s: Complex<f64> = a_k.iter().zip(tmp.iter()).map(|(&a, &t)| a * t).sum();
            fid.push(s);
            for i in 0..d {
                a_k[i] *= self.phases[i];
            }
        }

        fid
    }
}

impl std::fmt::Debug for EigenPropagator {
    // Hand-rolled Debug because V is a D×D complex matrix — we don't want to
    // dump thousands of entries by default. Expose the shape + dt only.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EigenPropagator")
            .field("dim", &self.v.nrows())
            .field("dt", &self.dt)
            .finish()
    }
}

impl Propagator for EigenPropagator {
    fn apply(&self, rho: &DensityMatrix) -> DensityMatrix {
        // Slow path: transform into the eigenbasis, multiply elementwise by
        // exp(−i(Eᵢ − Eⱼ)Δt), transform back. O(D³) per call (two matmuls).
        //
        // Exists for Propagator-trait satisfaction and correctness testing.
        // For real FID computation call [`Self::compute_fid`], which stays
        // in the eigenbasis for the whole N-step loop.
        let d = self.v.nrows();
        debug_assert_eq!(rho.nrows(), d);
        debug_assert_eq!(rho.ncols(), d);

        let mut rho_eig = &self.v_dagger * rho * &self.v;
        for i in 0..d {
            for j in 0..d {
                rho_eig[(i, j)] *= self.phases[i] * self.phases[j].conj();
            }
        }
        &self.v * rho_eig * &self.v_dagger
    }

    fn dt(&self) -> f64 {
        self.dt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fid::compute_fid;
    use crate::hamiltonian::{JCouplingH, SumH, ZeemanH};
    use crate::operator::total_m_minus;
    use crate::propagator::{DiagonalPropagator, MatrixPropagator};
    use crate::spin::{Isotope, Spin, SpinSystem};
    use crate::state::thermal_x_state;

    fn two_proton_system(shifts: [f64; 2]) -> SpinSystem {
        SpinSystem::new(
            vec![
                Spin::new(Isotope::H1, shifts[0]),
                Spin::new(Isotope::H1, shifts[1]),
            ],
            14.0954,
        )
    }

    fn three_proton_system() -> SpinSystem {
        SpinSystem::new(
            vec![
                Spin::new(Isotope::H1, 1.0),
                Spin::new(Isotope::H1, 4.0),
                Spin::new(Isotope::H1, 7.0),
            ],
            14.0954,
        )
    }

    // ---- construction / diagnostics ----

    #[test]
    fn eigenvalues_sum_to_trace_of_h() {
        // Tr(H) = Σ Eₖ for any diagonalizable matrix. A quick eigenvalue
        // sanity check that catches gross bugs in the diagonalization pipeline
        // (e.g. wrong triangle extracted, sign flip, dimension mismatch).
        let sys = two_proton_system([1.0, 5.0]);
        let h = SumH::new(vec![
            Box::new(ZeemanH::new(&sys)),
            Box::new(JCouplingH::new(&sys, &[(0, 1, 7.0)])),
        ]);
        let p = EigenPropagator::new(&h, 5e-6);

        let m = h.as_dense();
        let mut trace = 0.0_f64;
        for k in 0..m.nrows() {
            trace += m[(k, k)].re;
        }
        let eig_sum: f64 = p.eigenvalues().iter().sum();
        assert!((eig_sum - trace).abs() < 1e-6);
    }

    #[test]
    fn eigenvectors_span_orthonormal_basis() {
        // V† V = 𝟙 to machine precision for a real-symmetric H.
        let sys = three_proton_system();
        let h = SumH::new(vec![
            Box::new(ZeemanH::new(&sys)),
            Box::new(JCouplingH::new(
                &sys,
                &[(0, 1, 7.0), (1, 2, 5.0), (0, 2, 1.2)],
            )),
        ]);
        let p = EigenPropagator::new(&h, 1e-5);
        let v = p.eigenvectors();
        let prod = &v.adjoint() * v;
        let d = v.nrows();
        for i in 0..d {
            for j in 0..d {
                let expected = if i == j {
                    Complex::new(1.0, 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
                assert!(
                    (prod[(i, j)] - expected).norm() < 1e-10,
                    "V† V not identity at ({i}, {j}): {:?}",
                    prod[(i, j)]
                );
            }
        }
    }

    // ---- Propagator trait correctness (slow path) ----

    #[test]
    fn slow_apply_agrees_with_matrix_propagator_on_zeeman() {
        // Plain Zeeman (diagonal H): the eigen and matrix propagators must
        // give identical ρ(t) to machine precision for any number of steps.
        let sys = three_proton_system();
        let h = ZeemanH::new(&sys);
        let dt: f64 = 5e-5;
        let p_eig = EigenPropagator::new(&h, dt);
        let p_mat = MatrixPropagator::new(&h, dt);

        let mut rho_eig = thermal_x_state(&sys);
        let mut rho_mat = rho_eig.clone();
        for _ in 0..100 {
            rho_eig = p_eig.apply(&rho_eig);
            rho_mat = p_mat.apply(&rho_mat);
        }

        let max_mag = rho_mat.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
        let tol: f64 = 1e-9 * max_mag;
        for i in 0..rho_eig.nrows() {
            for j in 0..rho_eig.ncols() {
                let diff = rho_eig[(i, j)] - rho_mat[(i, j)];
                assert!(
                    diff.norm() < tol,
                    "disagreement at ({i}, {j}) after 100 steps: \
                     eig={:?}, mat={:?}, tol={tol:.3e}",
                    rho_eig[(i, j)],
                    rho_mat[(i, j)],
                );
            }
        }
    }

    #[test]
    fn slow_apply_agrees_with_matrix_propagator_on_j_coupled() {
        // Non-diagonal H (Zeeman + J): same agreement requirement. This is
        // the case where MatrixPropagator is needed and EigenPropagator is
        // designed to replace it — so the outputs must match.
        let sys = two_proton_system([3.0, 3.5]);
        let h = SumH::new(vec![
            Box::new(ZeemanH::new(&sys)),
            Box::new(JCouplingH::new(&sys, &[(0, 1, 7.0)])),
        ]);
        let dt: f64 = 5e-5;
        let p_eig = EigenPropagator::new(&h, dt);
        let p_mat = MatrixPropagator::new(&h, dt);

        let mut rho_eig = thermal_x_state(&sys);
        let mut rho_mat = rho_eig.clone();
        for _ in 0..200 {
            rho_eig = p_eig.apply(&rho_eig);
            rho_mat = p_mat.apply(&rho_mat);
        }

        let max_mag = rho_mat.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
        let tol: f64 = 1e-9 * max_mag;
        for i in 0..rho_eig.nrows() {
            for j in 0..rho_eig.ncols() {
                assert!(
                    (rho_eig[(i, j)] - rho_mat[(i, j)]).norm() < tol,
                    "disagreement at ({i}, {j}): eig={:?}, mat={:?}",
                    rho_eig[(i, j)],
                    rho_mat[(i, j)],
                );
            }
        }
    }

    // ---- fast compute_fid correctness ----

    #[test]
    fn fast_compute_fid_matches_slow_path_on_zeeman() {
        // The fast path (stay in eigenbasis) and the generic compute_fid
        // routed through EigenPropagator::apply must produce identical FIDs.
        let sys = three_proton_system();
        let h = ZeemanH::new(&sys);
        let dt: f64 = 1e-5;
        let n: usize = 512;
        let p = EigenPropagator::new(&h, dt);

        let rho0 = thermal_x_state(&sys);
        let obs = total_m_minus(&sys);

        let fid_fast = p.compute_fid(&rho0, &obs, n);
        let fid_slow = compute_fid(&p, &rho0, &obs, n);

        assert_eq!(fid_fast.len(), n);
        assert_eq!(fid_slow.len(), n);

        let max_mag = fid_slow.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
        let tol: f64 = 1e-8 * max_mag;
        for k in 0..n {
            let diff = fid_fast[k] - fid_slow[k];
            assert!(
                diff.norm() < tol,
                "FID disagreement at k={k}: fast={:?}, slow={:?}",
                fid_fast[k],
                fid_slow[k],
            );
        }
    }

    #[test]
    fn fast_compute_fid_matches_matrix_propagator_on_j_coupled() {
        // The real point: EigenPropagator::compute_fid must agree with
        // MatrixPropagator's compute_fid on a J-coupled system. This is the
        // correctness handshake that entitles us to use the fast path on
        // 10-spin molecules where MatrixPropagator is too slow to run.
        let sys = two_proton_system([3.0, 3.5]);
        let h = SumH::new(vec![
            Box::new(ZeemanH::new(&sys)),
            Box::new(JCouplingH::new(&sys, &[(0, 1, 7.0)])),
        ]);
        let dt: f64 = 1e-5;
        let n: usize = 256;

        let p_eig = EigenPropagator::new(&h, dt);
        let p_mat = MatrixPropagator::new(&h, dt);

        let rho0 = thermal_x_state(&sys);
        let obs = total_m_minus(&sys);

        let fid_eig = p_eig.compute_fid(&rho0, &obs, n);
        let fid_mat = compute_fid(&p_mat, &rho0, &obs, n);

        let max_mag = fid_mat.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
        let tol: f64 = 1e-7 * max_mag;
        for k in 0..n {
            let diff = fid_eig[k] - fid_mat[k];
            assert!(
                diff.norm() < tol,
                "FID disagreement at k={k}: eig={:?}, mat={:?}",
                fid_eig[k],
                fid_mat[k],
            );
        }
    }

    #[test]
    fn fast_compute_fid_agrees_with_diagonal_on_single_spin() {
        // Degenerate sanity check: single 1H at δ ppm. The DiagonalPropagator
        // is correct by independent derivation; the eigendecomposition is
        // trivial (H is already diagonal, V = 𝟙). Fast-path FID must match
        // the diagonal-propagator FID.
        let b0: f64 = 14.0954;
        let sys = SpinSystem::new(vec![Spin::new(Isotope::H1, 3.7)], b0);
        let h = ZeemanH::new(&sys);
        let dt: f64 = 5e-6;
        let n: usize = 128;

        let p_eig = EigenPropagator::new(&h, dt);
        let p_diag = DiagonalPropagator::new(&h, dt);

        let rho0 = thermal_x_state(&sys);
        let obs = total_m_minus(&sys);

        let fid_eig = p_eig.compute_fid(&rho0, &obs, n);
        let fid_diag = compute_fid(&p_diag, &rho0, &obs, n);

        let max_mag = fid_diag.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
        let tol: f64 = 1e-9 * max_mag;
        for k in 0..n {
            assert!(
                (fid_eig[k] - fid_diag[k]).norm() < tol,
                "k={k}: eig={:?}, diag={:?}",
                fid_eig[k],
                fid_diag[k],
            );
        }
    }

    // ---- input validation ----

    #[test]
    #[should_panic(expected = "complex entries")]
    fn construction_rejects_complex_hamiltonian() {
        // Hand-built Hamiltonian with an imaginary off-diagonal — i.e. not
        // real in the product-Iz basis. Construction must refuse rather than
        // silently discarding the imaginary parts.
        struct ComplexH {
            m: Operator,
        }
        impl Hamiltonian for ComplexH {
            fn dim(&self) -> usize {
                self.m.nrows()
            }
            fn as_dense(&self) -> Operator {
                self.m.clone()
            }
        }

        let mut m = Operator::zeros(2, 2);
        m[(0, 1)] = Complex::new(0.0, 1.0); // pure imaginary off-diagonal
        m[(1, 0)] = Complex::new(0.0, -1.0); // (Hermitian, still complex)
        let h = ComplexH { m };
        let _ = EigenPropagator::new(&h, 1e-5);
    }
}
