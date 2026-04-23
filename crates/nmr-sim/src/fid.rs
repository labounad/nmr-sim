//! Free induction decay (FID): the time-domain signal an NMR spectrometer
//! actually records.
//!
//! After the π/2 pulse, the bulk magnetization precesses around B₀ and induces
//! a voltage in the receiver coil. That voltage — sampled at regular intervals
//! — is the FID. Numerically, we model it as
//!
//! ```text
//! s(t_k) = Tr[ Ô · ρ(t_k) ]
//! ```
//!
//! where Ô is the detection operator and ρ(t_k) is the density matrix at
//! time k·Δt. For a standard 1D experiment we take Ô = M⁻ = Σᵢ Î₋,i so
//! that each spin contributes a signal oscillating as e^{-iΔωt}, placing
//! its peak at positive Δω in the spectrum (see
//! [`crate::operator::total_m_minus`] for why this sign convention).
//!
//! # What this module does not do
//!
//! No relaxation, no windowing, no pulse sequence beyond the implicit
//! π/2-y pulse baked into [`crate::state::thermal_x_state`]. This is a pure
//! deterministic Liouville–von Neumann integrator: every FID decays only
//! through dephasing between oscillators (T₂* from bandwidth mismatch),
//! not through any physical relaxation mechanism. Apodization / relaxation
//! will arrive in M4+.

use crate::operator::Operator;
use crate::propagator::Propagator;
use crate::state::DensityMatrix;
use num_complex::Complex;

/// Trace of a matrix product `Tr[A · B]`, computed without forming `A · B`.
///
/// Using the identity `Tr[A·B] = Σᵢⱼ A[i,j]·B[j,i]`, this reduces the cost
/// from O(D³) (dense matmul) to O(D²) (elementwise sum). Crucial for FID
/// computation, where we evaluate a trace at every one of ~8k time points.
///
/// # Panics
///
/// Panics if the shapes are incompatible for a product — i.e., if
/// `a.ncols() != b.nrows()` or `b.ncols() != a.nrows()` (the latter is
/// required for the product to be square, which is necessary for its trace
/// to exist).
pub fn trace_product(a: &Operator, b: &Operator) -> Complex<f64> {
    assert_eq!(
        a.ncols(),
        b.nrows(),
        "trace_product: A.cols ({}) must equal B.rows ({})",
        a.ncols(),
        b.nrows(),
    );
    assert_eq!(
        b.ncols(),
        a.nrows(),
        "trace_product: B.cols ({}) must equal A.rows ({}) for A·B to be square",
        b.ncols(),
        a.nrows(),
    );

    let mut acc = Complex::new(0.0, 0.0);
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            acc += a[(i, j)] * b[(j, i)];
        }
    }
    acc
}

/// Compute an FID by evolving ρ and sampling `Tr[observable · ρ]` at each step.
///
/// Given a propagator `U = exp(−iHΔt)` (via the [`Propagator`] trait), an
/// initial density matrix ρ(0), and a detection operator Ô, returns the
/// complex signal
///
/// ```text
/// s[k] = Tr[ Ô · ρ(k·Δt) ]       for k = 0, 1, ..., n_points − 1
/// ```
///
/// The first sample `s[0]` is taken *before* any propagation, so downstream
/// consumers should interpret the time axis as `t_k = k · propagator.dt()`.
///
/// # Why a generic `P: Propagator`?
///
/// So this function can be reused verbatim once we add
/// `KrylovPropagator`, `EigenbasisPropagator`, etc. in later milestones.
/// The caller chooses the propagation strategy; this function doesn't care.
///
/// # Complexity
///
/// `O(n_points · (propagate_cost + D²))`. For [`crate::propagator::DiagonalPropagator`]
/// the per-step propagate cost is O(D²), giving O(n_points · D²) overall.
pub fn compute_fid<P: Propagator>(
    propagator: &P,
    rho_initial: &DensityMatrix,
    observable: &Operator,
    n_points: usize,
) -> Vec<Complex<f64>> {
    assert_eq!(
        observable.nrows(),
        observable.ncols(),
        "observable must be square",
    );
    assert_eq!(
        rho_initial.nrows(),
        observable.nrows(),
        "ρ and observable dimensions must match: {} vs {}",
        rho_initial.nrows(),
        observable.nrows(),
    );

    let mut fid = Vec::with_capacity(n_points);
    let mut rho = rho_initial.clone();
    for _ in 0..n_points {
        fid.push(trace_product(observable, &rho));
        rho = propagator.apply(&rho);
    }
    fid
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamiltonian::ZeemanH;
    use crate::operator::{ix_at, total_m_minus};
    use crate::propagator::DiagonalPropagator;
    use crate::spin::{Isotope, Spin, SpinSystem};
    use crate::state::thermal_x_state;

    // ---- trace_product sanity ----

    #[test]
    fn trace_product_matches_explicit_matmul_on_small_case() {
        // Build two arbitrary 3x3 complex matrices; check our shortcut
        // equals the naive Tr[A·B] obtained by forming the product.
        let a = Operator::from_row_slice(
            3,
            3,
            &[
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 1.0),
                Complex::new(0.0, -1.0),
                Complex::new(3.0, 0.0),
                Complex::new(-1.0, 2.0),
                Complex::new(0.5, 0.5),
                Complex::new(1.0, 1.0),
                Complex::new(0.0, 0.0),
                Complex::new(-2.0, 0.0),
            ],
        );
        let b = Operator::from_row_slice(
            3,
            3,
            &[
                Complex::new(0.0, 1.0),
                Complex::new(1.0, 0.0),
                Complex::new(-1.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(0.0, -1.0),
                Complex::new(1.0, 1.0),
                Complex::new(-1.0, 2.0),
                Complex::new(0.5, 0.0),
                Complex::new(0.0, 3.0),
            ],
        );
        let ab = &a * &b;
        let naive_trace: Complex<f64> = (0..3).map(|i| ab[(i, i)]).sum();
        let got = trace_product(&a, &b);
        assert!(
            (got - naive_trace).norm() < 1e-12,
            "fast trace {got:?} ≠ naive trace {naive_trace:?}",
        );
    }

    #[test]
    #[should_panic(expected = "A.cols")]
    fn trace_product_panics_on_shape_mismatch() {
        let a = Operator::zeros(2, 3);
        let b = Operator::zeros(2, 2);
        let _ = trace_product(&a, &b);
    }

    // ---- compute_fid end-to-end ----

    #[test]
    fn fid_has_right_length_and_starts_with_zeroth_sample() {
        let sys = SpinSystem::new(vec![Spin::new(Isotope::H1, 1.0)], 14.0954);
        let h = ZeemanH::new(&sys);
        let p = DiagonalPropagator::new(&h, 5e-6);
        let rho0 = thermal_x_state(&sys);
        let obs = total_m_minus(&sys);
        let fid = compute_fid(&p, &rho0, &obs, 128);
        assert_eq!(fid.len(), 128);

        // First sample should be Tr[M⁻ · ρ(0)], no propagation applied yet.
        let expected0 = trace_product(&obs, &rho0);
        assert!((fid[0] - expected0).norm() < 1e-20);
    }

    #[test]
    fn single_spin_fid_oscillates_at_chemical_shift() {
        // For a single 1H at δ ppm with ρ(0) = γ·Îx and Ô = Î₋ = Îx − i·Îy,
        // the signal is
        //   s(t) = Tr[(Îx − i·Îy) · γ · (Îx cos(Δω t) + Îy sin(Δω t))]
        //        = γ · (Tr[Îx²] cos − i·Tr[Îy²] sin)      (cross terms vanish)
        //        = γ · (½) · (cos(Δω t) − i sin(Δω t))    (Tr[Îx²] = Tr[Îy²] = ½)
        //        = (γ/2) · e^{−iΔω t}
        let b0 = 14.0954;
        let delta_ppm = 5.0;
        let sys = SpinSystem::new(vec![Spin::new(Isotope::H1, delta_ppm)], b0);
        let h = ZeemanH::new(&sys);
        let dt = 5e-6;
        let n = 64;
        let p = DiagonalPropagator::new(&h, dt);
        let rho0 = thermal_x_state(&sys);
        let obs = total_m_minus(&sys);
        let fid = compute_fid(&p, &rho0, &obs, n);

        let dw = sys.spins[0].shift_angular(b0);
        let gamma = Isotope::H1.gamma();
        for (k, &s) in fid.iter().enumerate() {
            let t = k as f64 * dt;
            let expected = Complex::from_polar(gamma / 2.0, -dw * t);
            // Tolerance is relative to γ/2 ~ 10⁸. 1e-4 · γ gives ~10⁴ absolute.
            assert!(
                (s - expected).norm() < 1e-4 * gamma.abs(),
                "step {k}: t={t}, expected {expected:?}, got {s:?}",
            );
        }
    }

    #[test]
    fn zero_hamiltonian_fid_is_constant() {
        // H = 0 → ρ stationary → FID is a flat DC signal equal to Tr[Ô · ρ(0)].
        let sys = SpinSystem::new(
            vec![Spin::new(Isotope::H1, 0.0), Spin::new(Isotope::H1, 0.0)],
            14.0954,
        );
        let h = ZeemanH::new(&sys);
        let p = DiagonalPropagator::new(&h, 10e-6);
        let rho0 = thermal_x_state(&sys);
        let obs = total_m_minus(&sys);
        let fid = compute_fid(&p, &rho0, &obs, 32);

        let reference = fid[0];
        for (k, s) in fid.iter().enumerate() {
            assert!(
                (s - reference).norm() < 1e-6,
                "step {k}: expected {reference:?}, got {s:?}",
            );
        }
    }

    #[test]
    fn two_spin_fid_is_sum_of_two_chemical_shift_oscillators() {
        // Two non-interacting 1H's should produce a FID that's the sum of
        // their individual chemical-shift oscillations. (Without J-coupling,
        // our Zeeman Hamiltonian has no cross terms — the two spins are
        // independent oscillators.)
        //
        // Per-site amplitude: Tr[M⁻ · γᵢ·Îx,i] in the 4-dim 2-spin Hilbert
        // space = γᵢ · Tr[Îx,i²] = γᵢ · Tr[Îx²] · Tr[𝟙_other] = γᵢ · (1/2) · 2
        // = γᵢ. So for an N-spin system the factor generalizes to 2^(N−2)
        // (which recovers γ/2 only for N = 1; here N = 2 → factor 1).
        let b0 = 14.0954;
        let shifts = [1.5_f64, 7.3_f64];
        let sys = SpinSystem::new(
            vec![
                Spin::new(Isotope::H1, shifts[0]),
                Spin::new(Isotope::H1, shifts[1]),
            ],
            b0,
        );
        let h = ZeemanH::new(&sys);
        let dt = 8e-6;
        let n = 256;
        let p = DiagonalPropagator::new(&h, dt);
        let rho0 = thermal_x_state(&sys);
        let obs = total_m_minus(&sys);
        let fid = compute_fid(&p, &rho0, &obs, n);

        let gamma = Isotope::H1.gamma();
        let dw: Vec<f64> = sys.spins.iter().map(|s| s.shift_angular(b0)).collect();

        for (k, &s) in fid.iter().enumerate() {
            let t = k as f64 * dt;
            let expected: Complex<f64> =
                dw.iter().map(|&w| Complex::from_polar(gamma, -w * t)).sum();
            assert!(
                (s - expected).norm() < 1e-3 * gamma.abs(),
                "step {k}: expected {expected:?}, got {s:?}",
            );
        }
    }

    #[test]
    fn fid_trace_invariant_under_static_observable() {
        // Sanity check: Tr[𝟙 · ρ(t)] should equal Tr[ρ(0)] for all t under
        // unitary propagation, because the trace is invariant. This test
        // also confirms compute_fid correctly computes `Tr[Ô · ρ]` for an
        // observable that is not M⁻.
        let sys = SpinSystem::new(vec![Spin::new(Isotope::H1, 2.0)], 14.0954);
        let h = ZeemanH::new(&sys);
        let p = DiagonalPropagator::new(&h, 5e-6);
        let mut rho0 = thermal_x_state(&sys);
        // Add an identity piece so Tr[ρ₀] ≠ 0.
        let dim = sys.dim() as usize;
        for i in 0..dim {
            rho0[(i, i)] += Complex::new(1.0 / dim as f64, 0.0);
        }
        let identity = Operator::identity(dim, dim);
        let fid = compute_fid(&p, &rho0, &identity, 50);
        let tr0 = fid[0];
        for (k, s) in fid.iter().enumerate() {
            assert!(
                (s - tr0).norm() < 1e-10,
                "trace drifted at step {k}: {tr0:?} → {s:?}",
            );
        }
    }

    #[test]
    fn fid_observable_ix_gives_real_signal() {
        // Hermitian observables produce real-valued signals against a
        // Hermitian ρ. We suppress the complex exponential structure of M⁻
        // by picking Ô = Σᵢ Îx,i (Hermitian) instead; the FID should be
        // purely real up to roundoff.
        let sys = SpinSystem::new(
            vec![Spin::new(Isotope::H1, 1.0), Spin::new(Isotope::H1, 4.5)],
            14.0954,
        );
        let h = ZeemanH::new(&sys);
        let p = DiagonalPropagator::new(&h, 5e-6);
        let rho0 = thermal_x_state(&sys);

        // Build Σᵢ Îx,i directly from the lifted ops.
        let dim = sys.dim() as usize;
        let mut mx = Operator::zeros(dim, dim);
        for i in 0..sys.len() {
            mx += ix_at(&sys, i);
        }
        let fid = compute_fid(&p, &rho0, &mx, 100);
        for (k, s) in fid.iter().enumerate() {
            assert!(
                s.im.abs() < 1e-5 * Isotope::H1.gamma().abs(),
                "step {k}: imaginary part {} not ~0 (FID = {s:?})",
                s.im,
            );
        }
    }
}
