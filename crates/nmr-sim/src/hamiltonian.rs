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
//!
//! # Planned next
//!
//! - `JCouplingH`: isotropic scalar coupling Σᵢ<ⱼ 2π J_{ij} Îᵢ·Îⱼ. Non-diagonal
//!   in the product-Iz basis (flip-flop terms); will implement `as_dense` but
//!   not `try_as_diagonal`.
//! - `DipolarH`: direct dipolar coupling with orientation dependence.
//! - `RfPulseH`: time-dependent RF irradiation (requires extending the trait).

use crate::operator::{iz_at, Operator};
use crate::spin::SpinSystem;
use nalgebra::DVector;
use num_complex::Complex;

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
}
