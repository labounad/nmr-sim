//! Density matrices and initial states.
//!
//! A density matrix ρ is a Hermitian, positive-semidefinite, trace-1 operator
//! on the spin Hilbert space. For simulation we usually work with the
//! "deviation density matrix" — ρ minus its uniform 𝟙/D piece — because:
//!
//! 1. The 𝟙/D part commutes with every Hamiltonian, so it is stationary
//!    under propagation and contributes nothing to traceless observables.
//! 2. Dropping it eliminates a large constant that would otherwise dominate
//!    ρ's Frobenius norm and make numerical comparisons harder.
//!
//! The cost of dropping 𝟙/D is that ρ is no longer trace-normalized to 1
//! (instead it's traceless), and we lose information about absolute signal
//! size. We happily accept both: NMR spectra are always reported in
//! arbitrary/normalized units.
//!
//! # Units
//!
//! Deviation density matrices returned by this module carry a γᵢ-weighting
//! but no temperature (see `docs/architecture.md` on why T drops out of
//! relative intensities). Their numerical scale is therefore set by the raw
//! gyromagnetic ratios in rad·s⁻¹·T⁻¹, i.e. values on the order of 10⁸.
//! This is fine for f64 arithmetic and is transparent after the FFT, which
//! is homogeneous in the input.

use crate::operator::{ix_at, Operator};
use crate::spin::SpinSystem;
use num_complex::Complex;

/// A density matrix on a spin system's Hilbert space.
///
/// This is just an alias for [`Operator`]; it exists to document intent at
/// type signatures (a function taking `&DensityMatrix` wants ρ, not any old
/// operator). Validity (Hermiticity, trace, positivity) is not enforced by
/// the type — producers are responsible for constructing valid states.
pub type DensityMatrix = Operator;

/// Initial deviation density matrix for a 1D NMR experiment:
/// post-π/2-pulse, high-temperature equilibrium, γ-weighted.
///
/// Returns `ρ(0) = Σᵢ γᵢ · Îx,i`, i.e. every spin is in a coherent
/// transverse state with an amplitude proportional to its gyromagnetic ratio.
/// This is the state an NMR experiment actually produces after a perfect
/// hard π/2-y pulse on the thermal-equilibrium magnetization, *up to a
/// global prefactor of ℏB₀/kT that we drop because it only sets absolute
/// signal size*.
///
/// # Why γ-weighting?
///
/// The thermal equilibrium density matrix in the high-temperature limit is
///
/// ```text
/// ρ_eq ≈ 𝟙/D + (ℏB₀ / kT) · Σᵢ γᵢ · Îz,i
/// ```
///
/// A π/2-y pulse rotates Îz → Îx, leaving the Îx,i coefficients proportional
/// to γᵢ. Temperature cancels in any relative intensity, but γᵢ does not —
/// so a heteronuclear system (e.g. 1H + 13C) correctly shows 13C peaks
/// ~¼ the height of 1H peaks, tracking the ratio γ_C / γ_H.
///
/// # Complexity
///
/// O(N · D²) where N is the number of spins and D the Hilbert-space
/// dimension. For N ≫ 1 the full D × D storage is the bottleneck — we'll
/// switch to polyadic storage in a later milestone.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spin::{Isotope, Spin};

    #[test]
    fn thermal_x_state_is_hermitian() {
        // ρ = Σᵢ γᵢ Îx,i is a real linear combination of Hermitian operators,
        // so must itself be Hermitian.
        let sys = SpinSystem::new(
            vec![Spin::new(Isotope::H1, 1.0), Spin::new(Isotope::C13, 100.0)],
            14.0954,
        );
        let rho = thermal_x_state(&sys);
        for i in 0..rho.nrows() {
            for j in 0..rho.ncols() {
                let diff = rho[(i, j)] - rho[(j, i)].conj();
                assert!(diff.norm() < 1e-6, "not Hermitian at ({i},{j})");
            }
        }
    }

    #[test]
    fn thermal_x_state_is_traceless() {
        // Each Îx,i is traceless (eigenvalues ±1/2, summing to 0), so the
        // γᵢ-weighted sum is traceless too.
        let sys = SpinSystem::new(
            vec![Spin::new(Isotope::H1, 1.0), Spin::new(Isotope::H1, 2.0)],
            14.0954,
        );
        let rho = thermal_x_state(&sys);
        let tr: Complex<f64> = (0..rho.nrows()).map(|i| rho[(i, i)]).sum();
        assert!(tr.norm() < 1e-6, "trace = {tr:?} should be 0");
    }

    #[test]
    fn heteronuclear_amplitudes_are_gamma_weighted() {
        // For a 1H+13C system, the Frobenius norm contribution from each
        // site scales as |γ|. Build single-spin systems and compare — ρ for
        // a 1H should have ~γ_H/γ_C ≈ 4× the norm of ρ for 13C.
        let b0 = 14.0954;
        let sys_h = SpinSystem::new(vec![Spin::new(Isotope::H1, 0.0)], b0);
        let sys_c = SpinSystem::new(vec![Spin::new(Isotope::C13, 0.0)], b0);

        let rho_h = thermal_x_state(&sys_h);
        let rho_c = thermal_x_state(&sys_c);

        let norm_h: f64 = rho_h.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let norm_c: f64 = rho_c.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();

        let expected_ratio = Isotope::H1.gamma().abs() / Isotope::C13.gamma().abs();
        let actual_ratio = norm_h / norm_c;
        assert!(
            (actual_ratio - expected_ratio).abs() / expected_ratio < 1e-10,
            "γ-weighting wrong: expected ratio {expected_ratio}, got {actual_ratio}"
        );
    }
}
