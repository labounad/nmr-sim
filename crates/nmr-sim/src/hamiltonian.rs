//! Hamiltonians for spin dynamics.
//!
//! Provides the [`Hamiltonian`] trait and concrete implementations of the
//! Hamiltonian terms relevant to NMR. **All Hamiltonians are expressed in the
//! rotating frame (one frame per isotope) and in units of rad/s.** This is
//! not negotiable: working in the lab frame would put numerically enormous
//! Larmor frequencies on the diagonal and destroy the interesting physics to
//! roundoff. See `docs/architecture.md` for the convention.
//!
//! # Current implementations
//!
//! - [`ZeemanH`]: isotropic chemical-shift (rotating-frame Zeeman) Hamiltonian.
//!
//! # Planned next
//!
//! - `JCouplingH`: isotropic scalar coupling Σᵢ<ⱼ 2π J_{ij} Îᵢ·Îⱼ.
//! - `DipolarH`: direct dipolar coupling with orientation dependence.
//! - `RfPulseH`: time-dependent RF irradiation (requires extending the trait).

use crate::operator::{iz_at, Operator};
use crate::spin::SpinSystem;
use num_complex::Complex;

/// A Hamiltonian exposes its matrix representation in rad/s.
///
/// For static (time-independent) Hamiltonians, implementors typically store
/// the precomputed matrix and just hand out a reference. Time-dependent
/// Hamiltonians (future extensions) will supply a richer interface layered
/// on top of this trait.
pub trait Hamiltonian {
    /// The matrix representation of this Hamiltonian, in rad/s.
    fn matrix(&self) -> &Operator;

    /// Hilbert-space dimension. Defaults to the matrix's row count.
    fn dim(&self) -> usize {
        self.matrix().nrows()
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
/// on the diagonal. The result is diagonal in the product-Iz basis.
///
/// For a homonuclear system, this is the standard "offsets only" Zeeman
/// term. For heteronuclear systems, each isotope is implicitly in its own
/// rotating frame — a design we'll make explicit when J/dipolar couplings
/// between heteronuclei are added (they bring in the secular approximation).
#[derive(Debug, Clone)]
pub struct ZeemanH {
    matrix: Operator,
}

impl ZeemanH {
    /// Build the rotating-frame Zeeman Hamiltonian for `sys`.
    ///
    /// O(N · D²) in time and O(D²) in memory, where N is the number of
    /// spins and D = ∏ᵢ (2Iᵢ + 1) is the full Hilbert-space dimension.
    pub fn new(sys: &SpinSystem) -> Self {
        let dim = sys.dim() as usize;
        let mut matrix = Operator::zeros(dim, dim);
        for (i, spin) in sys.spins.iter().enumerate() {
            let omega = spin.shift_angular(sys.b0_tesla);
            // Skip spins at zero shift — their contribution is exactly zero
            // and the allocation/kronecker work is wasted. Not correctness-
            // critical; purely a small constant-factor win for sparse systems.
            if omega == 0.0 {
                continue;
            }
            let iz_i = iz_at(sys, i);
            matrix += iz_i.map(|x| x * Complex::new(omega, 0.0));
        }
        Self { matrix }
    }
}

impl Hamiltonian for ZeemanH {
    fn matrix(&self) -> &Operator {
        &self.matrix
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
        assert_eq!(h.matrix().shape(), (4, 4));
        for x in h.matrix().iter() {
            assert!(x.norm() < 1e-12);
        }
    }

    #[test]
    fn zeeman_is_diagonal_in_product_basis() {
        let sys = two_protons([1.0, 5.0]);
        let h = ZeemanH::new(&sys);
        let m = h.matrix();
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
        // H = H†, so elementwise we should have H[i,j] = conj(H[j,i]).
        let sys = two_protons([1.0, 5.0]);
        let h = ZeemanH::new(&sys);
        let m = h.matrix();
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
        let m = h.matrix();
        let tr: Complex<f64> = (0..m.nrows()).map(|i| m[(i, i)]).sum();
        assert!(tr.norm() < 1e-10, "trace = {tr:?} should be 0");
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
        let m = h.matrix();

        let expected = [
            0.5 * d1 + 0.5 * d2,
            0.5 * d1 - 0.5 * d2,
            -0.5 * d1 + 0.5 * d2,
            -0.5 * d1 - 0.5 * d2,
        ];
        for (i, &e) in expected.iter().enumerate() {
            assert!(
                (m[(i, i)].re - e).abs() < 1e-6,
                "diagonal[{i}]: got {}, expected {}",
                m[(i, i)].re,
                e,
            );
            assert!(m[(i, i)].im.abs() < 1e-12);
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
        let m = h.matrix();

        let expected = [
            0.5 * d_h + 0.5 * d_c,
            0.5 * d_h - 0.5 * d_c,
            -0.5 * d_h + 0.5 * d_c,
            -0.5 * d_h - 0.5 * d_c,
        ];
        for (i, &e) in expected.iter().enumerate() {
            assert!((m[(i, i)].re - e).abs() < 1e-6);
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
}
