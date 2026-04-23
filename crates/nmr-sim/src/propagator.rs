//! Time propagation of density matrices.
//!
//! The Liouville–von Neumann equation in the rotating frame is
//!
//! ```text
//! dρ/dt = −i [H, ρ]          ⇒          ρ(t + Δt) = U ρ(t) U†
//! ```
//!
//! where `U = exp(−iHΔt)` is the unitary propagator. This module packages the
//! computation of `U` and the act of applying it to a state into a small
//! trait-based API.
//!
//! # Available propagators
//!
//! - [`DiagonalPropagator`]: O(D²) apply, O(D) storage. Applies when the
//!   Hamiltonian advertises [`try_as_diagonal`](crate::Hamiltonian::try_as_diagonal).
//!   The fast path. Used automatically for [`ZeemanH`](crate::ZeemanH).
//! - [`MatrixPropagator`]: O(D³) apply, O(D²) storage. Universal fallback —
//!   takes any `Hamiltonian`, builds `U = exp(−iHΔt)` as a dense matrix via
//!   nalgebra's scaling-and-squaring Padé approximation, and applies it as a
//!   full triple product. Works correctly for any (finite) Hermitian H but
//!   scales poorly. Needed once J-coupling introduces off-diagonal terms in
//!   the product-Iz basis.
//!
//! More specializations are planned: `SparsePropagator` (sparse U, for
//! moderately off-diagonal systems), `KrylovPropagator` (matrix-free,
//! evaluates `exp(−iHΔt)·ρ` via Lanczos without ever building U), and
//! `RestrictedBasisPropagator` (state-space restriction à la Spinach).
//!
//! # Time step
//!
//! Nyquist–Shannon requires `Δt ≤ ‖H‖₂⁻¹` to avoid aliasing in the frequency
//! domain. For a rotating-frame Zeeman Hamiltonian with ≤ ~12 ppm spread at
//! 14 T, `‖H‖₂` is of order 2π · 7200 rad/s ≈ 4.5 × 10⁴ rad/s, so Δt should
//! be below ~22 μs. Callers pick Δt; we don't enforce Nyquist here, but
//! constructing a propagator with a Δt that violates it will produce a
//! spectrum that is silently wrong in the aliased bands.

use crate::hamiltonian::Hamiltonian;
use crate::operator::Operator;
use crate::state::DensityMatrix;
use nalgebra::DVector;
use num_complex::Complex;

/// A time-step operator U = exp(−iHΔt) that can be applied repeatedly to a
/// density matrix.
///
/// Concrete implementations differ in how they represent U internally — as a
/// full dense matrix, as a diagonal vector, as a sequence of small kron
/// factors, or lazily via Krylov iteration. They all expose the same user-
/// facing primitive: [`apply`](Propagator::apply) moves ρ forward by one Δt.
pub trait Propagator {
    /// Apply U · ρ · U† and return the result.
    ///
    /// The input is borrowed, not consumed, so callers can reuse ρ(0) in
    /// ensembles or starting-state sweeps.
    fn apply(&self, rho: &DensityMatrix) -> DensityMatrix;

    /// The Δt (in seconds) this propagator advances ρ by per call to
    /// [`apply`](Propagator::apply). Needed by downstream code (e.g. FID
    /// sampling) to build the time axis.
    fn dt(&self) -> f64;
}

/// Propagator specialized to diagonal Hamiltonians.
///
/// If H is diagonal with real entries `h_k`, then
///
/// ```text
/// U = exp(−iHΔt) = diag(exp(−i h_k Δt))
/// ```
///
/// and applying U · ρ · U† to a density matrix reduces to multiplying each
/// matrix element pointwise:
///
/// ```text
/// ρ'[i, j] = exp(−i(h_i − h_j) Δt) · ρ[i, j]
/// ```
///
/// This is O(D²) per step with zero matrix multiplications, and storage is
/// a single `DVector<Complex<f64>>` of length D — not a D×D dense matrix.
///
/// # When it applies
///
/// - Rotating-frame Zeeman Hamiltonians (always).
/// - Any Hamiltonian pre-rotated into its own eigenbasis.
///
/// It does **not** apply once J-couplings are turned on (those introduce
/// off-diagonal raising/lowering terms in the product-Iz basis). Use
/// [`MatrixPropagator`] there.
///
/// # Construction
///
/// The [`new`](DiagonalPropagator::new) constructor delegates to
/// [`Hamiltonian::try_as_diagonal`]; if the Hamiltonian refuses to advertise
/// diagonal storage, `new` panics rather than silently scanning a dense
/// matrix for diagonality. To build directly from a known-good diagonal,
/// use [`from_diagonal`](DiagonalPropagator::from_diagonal).
#[derive(Debug, Clone)]
pub struct DiagonalPropagator {
    /// Per-basis-state phase factors: `phases[k] = exp(−i h_k Δt)`.
    phases: DVector<Complex<f64>>,
    dt: f64,
}

impl DiagonalPropagator {
    /// Build a propagator from a Hamiltonian that advertises diagonal storage.
    ///
    /// # Panics
    ///
    /// Panics if `h.try_as_diagonal()` returns `None`. The correct response
    /// for a non-diagonal Hamiltonian is to use [`MatrixPropagator`] (or a
    /// future sparse / Krylov specialization) instead of this one.
    pub fn new<H: Hamiltonian + ?Sized>(h: &H, dt: f64) -> Self {
        let diag = h.try_as_diagonal().unwrap_or_else(|| {
            panic!(
                "DiagonalPropagator requires a Hamiltonian that advertises \
                 diagonal storage via `try_as_diagonal`; use MatrixPropagator \
                 (or a future sparse/Krylov specialization) for general H"
            )
        });
        Self::from_diagonal(diag, dt)
    }

    /// Build a propagator directly from a real diagonal `h_k` and a time
    /// step `dt`. This is the canonical constructor — [`new`](Self::new) is
    /// a thin adapter over this plus [`Hamiltonian::try_as_diagonal`].
    #[must_use]
    pub fn from_diagonal(diagonal: &DVector<f64>, dt: f64) -> Self {
        let phases = DVector::from_iterator(
            diagonal.len(),
            diagonal
                .iter()
                .map(|&h_k| Complex::from_polar(1.0, -h_k * dt)),
        );
        Self { phases, dt }
    }
}

impl Propagator for DiagonalPropagator {
    fn apply(&self, rho: &DensityMatrix) -> DensityMatrix {
        let n = rho.nrows();
        debug_assert_eq!(n, rho.ncols(), "ρ must be square");
        debug_assert_eq!(n, self.phases.len(), "ρ dim must match propagator dim");

        let mut result = rho.clone();
        // U ρ U† with U diagonal: each (i,j) picks up phase_i · conj(phase_j).
        for i in 0..n {
            for j in 0..n {
                result[(i, j)] *= self.phases[i] * self.phases[j].conj();
            }
        }
        result
    }

    fn dt(&self) -> f64 {
        self.dt
    }
}

/// Universal dense propagator. Builds `U = exp(−iHΔt)` and applies it as a
/// triple matrix product.
///
/// # How it works
///
/// We form the antihermitian matrix `A = −i H Δt` and compute `U = exp(A)`
/// via nalgebra's `Matrix::exp` method, which uses scaling-and-squaring with
/// Padé approximation. For Hermitian `H`, `A` is antihermitian and `exp(A)`
/// is unitary to machine precision. The cached adjoint `U†` is stored
/// alongside `U` to avoid recomputing it per [`apply`](Propagator::apply).
///
/// # When to use
///
/// When `Hamiltonian::try_as_diagonal` returns `None` — e.g. J-coupled
/// systems whose Hamiltonians carry off-diagonal flip-flop terms in the
/// product-Iz basis. Works for any size `D`, but the O(D³) apply cost and
/// O(D²) storage make this impractical above ~12 spin-1/2's (4096 dim).
///
/// For larger systems we'll need sparse, matrix-free, or restricted-basis
/// specializations. This one is the correctness-reference fallback.
#[derive(Debug, Clone)]
pub struct MatrixPropagator {
    /// The dense unitary U = exp(−iHΔt), size D×D.
    u: Operator,
    /// U† cached at construction.
    u_dagger: Operator,
    dt: f64,
}

impl MatrixPropagator {
    /// Build a propagator from any Hamiltonian by materializing its dense
    /// representation and exponentiating.
    pub fn new<H: Hamiltonian + ?Sized>(h: &H, dt: f64) -> Self {
        Self::from_matrix(&h.as_dense(), dt)
    }

    /// Build a propagator directly from an already-materialized dense
    /// Hamiltonian matrix. The caller is responsible for ensuring `h` is
    /// Hermitian in rad/s units.
    #[must_use]
    pub fn from_matrix(h: &Operator, dt: f64) -> Self {
        assert_eq!(h.nrows(), h.ncols(), "Hamiltonian must be square");

        // A = −i H Δt. For Hermitian H, A is antihermitian, so exp(A) is
        // unitary. nalgebra's `exp` uses scaling-and-squaring + Padé, which
        // preserves unitarity to machine precision for antihermitian input.
        let minus_i_dt = Complex::new(0.0, -dt);
        let antiherm: Operator = h.map(|x| x * minus_i_dt);
        let u = antiherm.exp();
        let u_dagger = u.adjoint();
        Self { u, u_dagger, dt }
    }

    /// Borrow the stored unitary U. Useful for diagnostics and for callers
    /// that want to chain propagators manually.
    pub fn unitary(&self) -> &Operator {
        &self.u
    }
}

impl Propagator for MatrixPropagator {
    fn apply(&self, rho: &DensityMatrix) -> DensityMatrix {
        // ρ' = U ρ U†. nalgebra's `*` on DMatrix returns an owned matrix.
        &self.u * rho * &self.u_dagger
    }

    fn dt(&self) -> f64 {
        self.dt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamiltonian::ZeemanH;
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

    #[test]
    fn zero_hamiltonian_gives_identity_propagator() {
        // H = 0 → U = 𝟙 → ρ unchanged.
        let sys = two_proton_system([0.0, 0.0]);
        let h = ZeemanH::new(&sys);
        let p = DiagonalPropagator::new(&h, 10e-6);
        let rho = thermal_x_state(&sys);
        let rho_next = p.apply(&rho);
        for i in 0..rho.nrows() {
            for j in 0..rho.ncols() {
                let diff = rho[(i, j)] - rho_next[(i, j)];
                assert!(diff.norm() < 1e-20, "ρ changed at ({i},{j}): {diff:?}");
            }
        }
    }

    #[test]
    fn propagation_preserves_hermiticity() {
        // U ρ U† is Hermitian iff ρ is.
        let sys = two_proton_system([1.0, 5.0]);
        let h = ZeemanH::new(&sys);
        let p = DiagonalPropagator::new(&h, 10e-6);
        let mut rho = thermal_x_state(&sys);
        for _ in 0..50 {
            rho = p.apply(&rho);
        }
        for i in 0..rho.nrows() {
            for j in 0..rho.ncols() {
                let diff = rho[(i, j)] - rho[(j, i)].conj();
                assert!(
                    diff.norm() < 1e-6,
                    "not Hermitian at ({i},{j}) after evolution"
                );
            }
        }
    }

    #[test]
    fn propagation_preserves_trace() {
        // For any unitary U, Tr[U ρ U†] = Tr[ρ]. Our thermal state is
        // traceless (0), but let's verify the invariant on a non-traceless
        // state too. We'll add a small multiple of the identity.
        let sys = two_proton_system([2.0, 4.0]);
        let h = ZeemanH::new(&sys);
        let p = DiagonalPropagator::new(&h, 5e-6);
        let dim = sys.dim() as usize;
        let mut rho = thermal_x_state(&sys);
        // Add 𝟙/D so the trace is 1 (the "full" density matrix with the
        // identity piece restored).
        for i in 0..dim {
            rho[(i, i)] += Complex::new(1.0 / dim as f64, 0.0);
        }
        let tr0: Complex<f64> = (0..dim).map(|i| rho[(i, i)]).sum();

        for _ in 0..200 {
            rho = p.apply(&rho);
        }
        let tr1: Complex<f64> = (0..dim).map(|i| rho[(i, i)]).sum();
        assert!(
            (tr0 - tr1).norm() < 1e-10,
            "trace drifted: {tr0:?} → {tr1:?}"
        );
    }

    #[test]
    fn single_spin_coherence_oscillates_at_chemical_shift() {
        // For a single 1H at δ ppm, ρ(0) = γ·Îx evolves under H = Δω·Îz as
        //   ρ(t) = γ · (Îx cos(Δω t) + Îy sin(Δω t))
        // so the matrix element ρ[0,1](t) = γ · (½ cos − i·½ sin) = (γ/2) e^{-iΔωt}
        // (with our "first spin varies slowest" / Îz ordering, the [0,1]
        // element corresponds to the ↑↓ coherence of a single spin: m=+½ to m=−½).
        let b0 = 14.0954;
        let delta = 5.0; // ppm
        let sys = SpinSystem::new(vec![Spin::new(Isotope::H1, delta)], b0);
        let h = ZeemanH::new(&sys);
        let dt = 5e-6;
        let p = DiagonalPropagator::new(&h, dt);
        let rho0 = thermal_x_state(&sys);
        let dw = sys.spins[0].shift_angular(b0);
        let gamma = Isotope::H1.gamma();

        let mut rho = rho0.clone();
        for k in 0..20 {
            let t = k as f64 * dt;
            let expected = Complex::from_polar(gamma / 2.0, -dw * t);
            let got = rho[(0, 1)];
            assert!(
                (got - expected).norm() < 1e-4 * gamma,
                "step {k}: expected {expected:?}, got {got:?}"
            );
            rho = p.apply(&rho);
        }
    }

    #[test]
    #[should_panic(expected = "diagonal storage")]
    fn diagonal_propagator_rejects_non_diagonal_hamiltonian() {
        // Build a "Hamiltonian" that returns None from try_as_diagonal. The
        // concrete matrix doesn't matter — DiagonalPropagator should refuse
        // based on the advertised storage alone.
        struct FakeH {
            m: Operator,
        }
        impl Hamiltonian for FakeH {
            fn dim(&self) -> usize {
                self.m.nrows()
            }
            fn as_dense(&self) -> Operator {
                self.m.clone()
            }
            // try_as_diagonal defaults to None.
        }

        let sys = two_proton_system([1.0, 3.0]);
        let fake = FakeH {
            m: ZeemanH::new(&sys).as_dense(),
        };
        let _ = DiagonalPropagator::new(&fake, 5e-6);
    }

    // ---- MatrixPropagator tests ----

    #[test]
    fn matrix_propagator_agrees_with_diagonal_on_zeeman() {
        // For a ZeemanH, DiagonalPropagator and MatrixPropagator should
        // produce identical (to machine precision) evolved density matrices.
        // This is our cross-backend correctness check: the fast path and
        // the universal fallback must agree on overlapping inputs.
        let sys = SpinSystem::new(
            vec![
                Spin::new(Isotope::H1, 1.0),
                Spin::new(Isotope::H1, 4.0),
                Spin::new(Isotope::H1, 7.2),
            ],
            14.0954,
        );
        let h = ZeemanH::new(&sys);
        let dt = 5e-5;

        let p_diag = DiagonalPropagator::new(&h, dt);
        let p_mat = MatrixPropagator::new(&h, dt);

        let mut rho_diag = thermal_x_state(&sys);
        let mut rho_mat = rho_diag.clone();

        // Evolve both for a non-trivial number of steps and compare.
        for _ in 0..100 {
            rho_diag = p_diag.apply(&rho_diag);
            rho_mat = p_mat.apply(&rho_mat);
        }

        // Tolerance reasoning: matrix entries here have magnitude ~γ·½ ≈
        // 1.3·10⁸ rad/s (γ_H times the Îx scale factor). 100 matmuls on an
        // 8×8 matrix accumulate O(100·8·ε_mach) ≈ 2·10⁻¹³ relative drift
        // per element. A relative tolerance of 10⁻¹¹ is comfortably above
        // that but still rejects any real physics discrepancy.
        let max_mag = rho_diag.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
        let tol = 1e-11 * max_mag;
        for i in 0..rho_diag.nrows() {
            for j in 0..rho_diag.ncols() {
                let diff = rho_diag[(i, j)] - rho_mat[(i, j)];
                assert!(
                    diff.norm() < tol,
                    "backends disagree at ({i},{j}) after 100 steps: \
                     diag={:?}, mat={:?}, diff={diff:?}, tol={tol:.3e}",
                    rho_diag[(i, j)],
                    rho_mat[(i, j)],
                );
            }
        }
    }

    #[test]
    fn matrix_propagator_unitary_is_unitary() {
        // Sanity check: U U† = 𝟙 to machine precision for any Hermitian H.
        let sys = two_proton_system([1.0, 5.0]);
        let h = ZeemanH::new(&sys);
        let p = MatrixPropagator::new(&h, 5e-6);
        let u = p.unitary();
        let id = u * u.adjoint();
        let dim = u.nrows();
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j {
                    Complex::new(1.0, 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
                assert!(
                    (id[(i, j)] - expected).norm() < 1e-10,
                    "U U† off at ({i},{j}) = {:?}",
                    id[(i, j)],
                );
            }
        }
    }

    #[test]
    fn matrix_propagator_preserves_hermiticity_and_trace() {
        // Same invariants as DiagonalPropagator, but driven through the
        // dense matrix-exponential path.
        let sys = two_proton_system([2.0, 4.0]);
        let h = ZeemanH::new(&sys);
        let p = MatrixPropagator::new(&h, 5e-6);
        let dim = sys.dim() as usize;
        let mut rho = thermal_x_state(&sys);
        for i in 0..dim {
            rho[(i, i)] += Complex::new(1.0 / dim as f64, 0.0);
        }
        let tr0: Complex<f64> = (0..dim).map(|i| rho[(i, i)]).sum();

        for _ in 0..200 {
            rho = p.apply(&rho);
        }
        let tr1: Complex<f64> = (0..dim).map(|i| rho[(i, i)]).sum();
        assert!((tr0 - tr1).norm() < 1e-10);
        for i in 0..dim {
            for j in 0..dim {
                let diff = rho[(i, j)] - rho[(j, i)].conj();
                assert!(diff.norm() < 1e-6);
            }
        }
    }
}
