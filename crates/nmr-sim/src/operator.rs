//! Spin angular-momentum operators.
//!
//! Provides the standard single-spin operators Îx, Îy, Îz, Î+, Î− for arbitrary
//! spin-I, and a `lift` routine that embeds them into the full product Hilbert
//! space of a multi-spin system via Kronecker products.
//!
//! # Units and conventions
//!
//! All operators are dimensionless (in units where ℏ = 1). For spin-1/2 you
//! recover the Pauli matrices divided by two:
//!
//! - Îx = (1/2) · σₓ
//! - Îy = (1/2) · σᵧ
//! - Îz = (1/2) · σ_z
//!
//! # Basis ordering
//!
//! For a spin-I nucleus, the Zeeman basis is |I,m⟩ for m = I, I−1, ..., −I.
//! We store these in *descending* order of m: row/column 0 corresponds to
//! m = +I, row/column (2I) corresponds to m = −I. So for spin-1/2:
//!
//! ```text
//! index 0 ↔ m = +1/2 (spin "up")
//! index 1 ↔ m = −1/2 (spin "down")
//! ```
//!
//! # Multi-spin basis ordering
//!
//! For a system of N spins, basis states are tensor products
//! |m₁⟩ ⊗ |m₂⟩ ⊗ ... ⊗ |m_N⟩, and we use the convention matching
//! `nalgebra::Matrix::kronecker`: the *first* spin's index varies *slowest*.
//! So for two spin-1/2's (indexing with ↑ = +1/2, ↓ = −1/2):
//!
//! ```text
//! basis index 0: |↑↑⟩
//! basis index 1: |↑↓⟩
//! basis index 2: |↓↑⟩
//! basis index 3: |↓↓⟩
//! ```
//!
//! This matters for every multi-spin matrix you inspect — stick to this
//! convention everywhere and the Kronecker math stays consistent.

use crate::spin::SpinSystem;
use nalgebra::DMatrix;
use num_complex::Complex;

/// An operator on a (possibly multi-spin) Hilbert space: complex dense matrix.
///
/// We use `DMatrix` (dynamically-sized) rather than the statically-sized
/// `Matrix2`/`Matrix3`/... because spin-system dimensions are set at runtime.
pub type Operator = DMatrix<Complex<f64>>;

// ============================================================================
// Single-spin operators for arbitrary spin-I
// ============================================================================

/// Îz for a single spin-I, where `two_i` = 2I.
///
/// Returns a (2I+1) × (2I+1) diagonal matrix with entries m = I, I−1, ..., −I.
pub fn iz(two_i: u32) -> Operator {
    let n = (two_i + 1) as usize;
    let i_val = two_i as f64 / 2.0;
    let mut m = Operator::zeros(n, n);
    for k in 0..n {
        // index k corresponds to m = I − k
        m[(k, k)] = Complex::new(i_val - k as f64, 0.0);
    }
    m
}

/// Raising operator Î+ for a single spin-I.
///
/// Î+|I,m⟩ = √(I(I+1) − m(m+1)) · |I,m+1⟩
///
/// In matrix form, this puts nonzero entries on the superdiagonal:
/// (Î+)_{k, k+1} where row k corresponds to m' = I−k (= m+1) and column k+1
/// corresponds to m = I−k−1.
pub fn iplus(two_i: u32) -> Operator {
    let n = (two_i + 1) as usize;
    let i_val = two_i as f64 / 2.0;
    let mut m = Operator::zeros(n, n);
    for col in 1..n {
        // Column `col` is the ket |I,m⟩ with m = I − col.
        let m_col = i_val - col as f64;
        // Matrix element ⟨I,m+1|Î+|I,m⟩ = √(I(I+1) − m(m+1))
        let coeff = (i_val * (i_val + 1.0) - m_col * (m_col + 1.0)).sqrt();
        m[(col - 1, col)] = Complex::new(coeff, 0.0);
    }
    m
}

/// Lowering operator Î− = (Î+)†.
///
/// Î−|I,m⟩ = √(I(I+1) − m(m−1)) · |I,m−1⟩
pub fn iminus(two_i: u32) -> Operator {
    iplus(two_i).adjoint()
}

/// Îx = (Î+ + Î−) / 2.
pub fn ix(two_i: u32) -> Operator {
    let half = Complex::new(0.5, 0.0);
    (iplus(two_i) + iminus(two_i)).map(|x| x * half)
}

/// Îy = (Î+ − Î−) / (2i).
///
/// The `2i` in the denominator becomes −i/2 in the numerator
/// (multiplying top and bottom by −i/−i):
/// 1 / (2i) = −i / 2.
pub fn iy(two_i: u32) -> Operator {
    let minus_half_i = Complex::new(0.0, -0.5);
    (iplus(two_i) - iminus(two_i)).map(|x| x * minus_half_i)
}

// ============================================================================
// Multi-spin: Kronecker-product lifts
// ============================================================================

/// Lift a single-spin operator `op` into the full product Hilbert space
/// defined by `dims`, placing `op` at site `site` and identity everywhere else.
///
/// `dims[j]` is the Hilbert-space dimension (multiplicity) of spin j.
///
/// # Panics
///
/// - If `site >= dims.len()`.
/// - If `op`'s dimension does not match `dims[site]`.
/// - If `op` is not square.
pub fn lift(dims: &[usize], op: &Operator, site: usize) -> Operator {
    assert!(
        site < dims.len(),
        "site {site} out of bounds for system with {} spins",
        dims.len(),
    );
    assert_eq!(op.nrows(), op.ncols(), "operator must be square");
    assert_eq!(
        op.nrows(),
        dims[site],
        "operator dimension {} does not match site {site} dimension {}",
        op.nrows(),
        dims[site],
    );

    // Start with the 1x1 identity and fold in one factor per site. For the
    // target site we Kronecker-multiply in `op`; for every other site we use
    // the identity on that site's Hilbert space. The Kronecker ordering here
    // (result ⊗ factor, accumulating left-to-right) gives the "first spin
    // varies slowest" convention documented at the top of the module.
    let mut result = Operator::identity(1, 1);
    for (j, &d) in dims.iter().enumerate() {
        let factor = if j == site {
            op.clone()
        } else {
            Operator::identity(d, d)
        };
        result = result.kronecker(&factor);
    }
    result
}

/// Per-site Hilbert-space dimensions of a spin system.
fn site_dims(sys: &SpinSystem) -> Vec<usize> {
    sys.spins
        .iter()
        .map(|s| s.isotope.multiplicity() as usize)
        .collect()
}

/// Îz of spin `site` in `sys`, lifted to the full product space.
pub fn iz_at(sys: &SpinSystem, site: usize) -> Operator {
    let dims = site_dims(sys);
    let op = iz(sys.spins[site].isotope.two_i());
    lift(&dims, &op, site)
}

/// Îx of spin `site` in `sys`, lifted to the full product space.
pub fn ix_at(sys: &SpinSystem, site: usize) -> Operator {
    let dims = site_dims(sys);
    let op = ix(sys.spins[site].isotope.two_i());
    lift(&dims, &op, site)
}

/// Îy of spin `site` in `sys`, lifted to the full product space.
pub fn iy_at(sys: &SpinSystem, site: usize) -> Operator {
    let dims = site_dims(sys);
    let op = iy(sys.spins[site].isotope.two_i());
    lift(&dims, &op, site)
}

/// Î+ of spin `site` in `sys`, lifted to the full product space.
pub fn iplus_at(sys: &SpinSystem, site: usize) -> Operator {
    let dims = site_dims(sys);
    let op = iplus(sys.spins[site].isotope.two_i());
    lift(&dims, &op, site)
}

/// Î− of spin `site` in `sys`, lifted to the full product space.
pub fn iminus_at(sys: &SpinSystem, site: usize) -> Operator {
    let dims = site_dims(sys);
    let op = iminus(sys.spins[site].isotope.two_i());
    lift(&dims, &op, site)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spin::{Isotope, Spin};

    /// Frobenius-ish equality for operators: elementwise |a−b| < tol.
    fn approx_eq(a: &Operator, b: &Operator, tol: f64) -> bool {
        if a.shape() != b.shape() {
            return false;
        }
        (a - b).iter().all(|x| x.norm() < tol)
    }

    // ---- Single-spin: spin-1/2 ----

    #[test]
    fn spin_half_iz_is_diag_half_minus_half() {
        let m = iz(1);
        assert_eq!(m.shape(), (2, 2));
        assert!((m[(0, 0)] - Complex::new(0.5, 0.0)).norm() < 1e-12);
        assert!((m[(1, 1)] - Complex::new(-0.5, 0.0)).norm() < 1e-12);
        assert!(m[(0, 1)].norm() < 1e-12);
        assert!(m[(1, 0)].norm() < 1e-12);
    }

    #[test]
    fn spin_half_ix_is_half_sigma_x() {
        // Ix = (1/2) * [[0, 1], [1, 0]]
        let expected = Operator::from_row_slice(
            2,
            2,
            &[
                Complex::new(0.0, 0.0),
                Complex::new(0.5, 0.0),
                Complex::new(0.5, 0.0),
                Complex::new(0.0, 0.0),
            ],
        );
        assert!(approx_eq(&ix(1), &expected, 1e-12));
    }

    #[test]
    fn spin_half_iy_is_half_sigma_y() {
        // Iy = (1/2) * [[0, -i], [i, 0]]
        let expected = Operator::from_row_slice(
            2,
            2,
            &[
                Complex::new(0.0, 0.0),
                Complex::new(0.0, -0.5),
                Complex::new(0.0, 0.5),
                Complex::new(0.0, 0.0),
            ],
        );
        assert!(approx_eq(&iy(1), &expected, 1e-12));
    }

    // ---- Single-spin: spin-1 ----

    #[test]
    fn spin_one_iz_is_diag_one_zero_minus_one() {
        let m = iz(2);
        assert_eq!(m.shape(), (3, 3));
        assert!((m[(0, 0)] - Complex::new(1.0, 0.0)).norm() < 1e-12);
        assert!((m[(1, 1)] - Complex::new(0.0, 0.0)).norm() < 1e-12);
        assert!((m[(2, 2)] - Complex::new(-1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn spin_one_iplus_has_sqrt_two_offdiagonals() {
        // (I+)_{0,1} and (I+)_{1,2} should both be √2
        let m = iplus(2);
        let sqrt2 = 2.0_f64.sqrt();
        assert!((m[(0, 1)] - Complex::new(sqrt2, 0.0)).norm() < 1e-12);
        assert!((m[(1, 2)] - Complex::new(sqrt2, 0.0)).norm() < 1e-12);
        // All other entries zero
        assert!(m[(0, 0)].norm() < 1e-12);
        assert!(m[(0, 2)].norm() < 1e-12);
        assert!(m[(1, 0)].norm() < 1e-12);
        assert!(m[(1, 1)].norm() < 1e-12);
        assert!(m[(2, 0)].norm() < 1e-12);
        assert!(m[(2, 1)].norm() < 1e-12);
        assert!(m[(2, 2)].norm() < 1e-12);
    }

    // ---- Algebraic identities ----

    #[test]
    fn commutator_ix_iy_equals_i_iz_spin_half() {
        // [Îx, Îy] = i · Îz
        let ix_ = ix(1);
        let iy_ = iy(1);
        let iz_ = iz(1);
        let lhs = &ix_ * &iy_ - &iy_ * &ix_;
        let i = Complex::new(0.0, 1.0);
        let rhs = iz_.map(|x| x * i);
        assert!(approx_eq(&lhs, &rhs, 1e-12));
    }

    #[test]
    fn commutator_ix_iy_equals_i_iz_spin_one() {
        let ix_ = ix(2);
        let iy_ = iy(2);
        let iz_ = iz(2);
        let lhs = &ix_ * &iy_ - &iy_ * &ix_;
        let i = Complex::new(0.0, 1.0);
        let rhs = iz_.map(|x| x * i);
        assert!(approx_eq(&lhs, &rhs, 1e-12));
    }

    #[test]
    fn casimir_i_squared_equals_i_i_plus_one_times_identity() {
        // Îx² + Îy² + Îz² = I(I+1) · 𝟙 — the SU(2) Casimir invariant.
        // True for every irreducible rep. If it ever fails, something is
        // fundamentally wrong with our operator construction.
        for two_i in [1u32, 2, 3, 4, 5] {
            let ix_ = ix(two_i);
            let iy_ = iy(two_i);
            let iz_ = iz(two_i);
            let i_sq = &ix_ * &ix_ + &iy_ * &iy_ + &iz_ * &iz_;
            let i_val = two_i as f64 / 2.0;
            let n = (two_i + 1) as usize;
            let expected =
                Operator::identity(n, n).map(|x| x * Complex::new(i_val * (i_val + 1.0), 0.0));
            assert!(
                approx_eq(&i_sq, &expected, 1e-10),
                "Casimir identity failed for two_i = {two_i}",
            );
        }
    }

    // ---- Multi-spin lifts ----

    fn two_proton_system() -> SpinSystem {
        SpinSystem::new(
            vec![Spin::new(Isotope::H1, 0.0), Spin::new(Isotope::H1, 1.0)],
            14.0954,
        )
    }

    #[test]
    fn iz_at_site_zero_two_protons() {
        // Iz of spin 0 in a 2-spin-1/2 system:
        //   Iz_0 = Îz ⊗ 𝟙 = diag(+1/2, +1/2, −1/2, −1/2)
        let op = iz_at(&two_proton_system(), 0);
        assert_eq!(op.shape(), (4, 4));
        let diag: Vec<f64> = (0..4).map(|i| op[(i, i)].re).collect();
        assert_eq!(diag, vec![0.5, 0.5, -0.5, -0.5]);
    }

    #[test]
    fn iz_at_site_one_two_protons() {
        // Iz of spin 1 in a 2-spin-1/2 system:
        //   Iz_1 = 𝟙 ⊗ Îz = diag(+1/2, −1/2, +1/2, −1/2)
        let op = iz_at(&two_proton_system(), 1);
        let diag: Vec<f64> = (0..4).map(|i| op[(i, i)].re).collect();
        assert_eq!(diag, vec![0.5, -0.5, 0.5, -0.5]);
    }

    #[test]
    fn different_site_operators_commute() {
        // [Iz_0, Ix_1] = 0 — single-spin operators on different sites
        // always commute, since they act on different tensor factors.
        let sys = two_proton_system();
        let a = iz_at(&sys, 0);
        let b = ix_at(&sys, 1);
        let commutator = &a * &b - &b * &a;
        let zero = Operator::zeros(4, 4);
        assert!(approx_eq(&commutator, &zero, 1e-12));
    }

    #[test]
    fn lift_preserves_commutator_on_same_site() {
        // [Ix_0, Iy_0] should still equal i · Iz_0 after lifting — the
        // algebra doesn't care what space we embed into.
        let sys = two_proton_system();
        let ix0 = ix_at(&sys, 0);
        let iy0 = iy_at(&sys, 0);
        let iz0 = iz_at(&sys, 0);
        let lhs = &ix0 * &iy0 - &iy0 * &ix0;
        let i = Complex::new(0.0, 1.0);
        let rhs = iz0.map(|x| x * i);
        assert!(approx_eq(&lhs, &rhs, 1e-12));
    }

    #[test]
    fn lift_mixed_isotope_system_has_correct_dimension() {
        // 1H (dim 2) + 2H (dim 3) → product dimension 6.
        let sys = SpinSystem::new(
            vec![Spin::new(Isotope::H1, 0.0), Spin::new(Isotope::H2, 0.0)],
            14.0954,
        );
        let op = iz_at(&sys, 0);
        assert_eq!(op.shape(), (6, 6));
        // Diagonal entries of Iz_0 should be three copies of +1/2 followed
        // by three copies of -1/2 (first spin varies slowest).
        let diag: Vec<f64> = (0..6).map(|i| op[(i, i)].re).collect();
        assert_eq!(diag, vec![0.5, 0.5, 0.5, -0.5, -0.5, -0.5]);
    }
}
