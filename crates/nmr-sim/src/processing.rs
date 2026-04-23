//! Post-FID, pre-FFT signal processing: apodization and zero-filling.
//!
//! These are the two standard transformations NMR processing applies between
//! the raw FID and the displayed spectrum. They live here as free functions
//! so callers compose them explicitly:
//!
//! ```
//! use nmr_sim::{apodize_exponential, zero_fill, DiscreteSpectrum};
//! use num_complex::Complex;
//!
//! let mut fid = vec![Complex::new(1.0, 0.0); 16];
//! let dt: f64 = 1.0 / 20_000.0;
//! apodize_exponential(&mut fid, dt, 1.0); // 1 Hz line broadening
//! let padded = zero_fill(fid, 4);          // 4× frequency-domain interpolation
//! let _spec = DiscreteSpectrum::from_fid(&padded, dt);
//! ```
//!
//! # Why apodize?
//!
//! An undamped FID truncated at `N` samples Fourier-transforms to a sinc —
//! a narrow main lobe plus slowly-decaying side-lobes. In magnitude mode those
//! side-lobes fold up as positive ringing ("fat tails"). Multiplying the FID
//! by `exp(-π · lb · t)` turns the sinc into a Lorentzian of absorption-mode
//! FWHM `lb` Hz, which is much cleaner to look at and analyze.
//!
//! Once real T₂ relaxation physics arrives (future milestone), the FID from
//! [`crate::fid::compute_fid`] will decay on its own and callers can set
//! `lb_hz = 0.0` (or skip this call). Until then, exponential apodization is
//! the phenomenological stand-in — equivalent to a uniform T₂ across all
//! transitions with T₂ = 1 / (π · lb_hz).
//!
//! # Why zero-fill?
//!
//! FFT bin width is `Δf = 1 / (N · dt)`. When the underlying line is narrower
//! than one bin you cannot see its shape — only a single tall sample with
//! straight lines to the neighbors. Zero-filling the FID by an integer factor
//! `k` makes the FFT emit `k · N` bins, each `Δf / k` Hz wide, covering the
//! same ±Nyquist window. The underlying information is unchanged (zero-filling
//! is frequency-domain interpolation, not super-resolution), but the rendered
//! lineshape becomes smooth.

use num_complex::Complex;
use std::f64::consts::PI;

/// Apply exponential apodization (line broadening) to an FID in place.
///
/// Multiplies each sample by `exp(-π · lb_hz · t)` where `t = k · dt` is the
/// time of sample `k`. In the frequency domain this convolves the spectrum
/// with a Lorentzian of absorption-mode FWHM `lb_hz` Hz (magnitude-mode FWHM
/// is √3 times larger).
///
/// A rule of thumb for 1H high-resolution NMR is `lb_hz = 1.0`: large enough
/// that peaks span several FFT bins and sinc side-lobe ringing is suppressed,
/// small enough that J-splittings down to ~1.5 Hz remain resolvable.
///
/// `lb_hz == 0.0` is a no-op; negative values are rejected because they would
/// exponentially *grow* the FID and break the FFT amplitude scale.
///
/// # Panics
///
/// Panics if `lb_hz < 0.0` or `dt <= 0.0`.
pub fn apodize_exponential(fid: &mut [Complex<f64>], dt: f64, lb_hz: f64) {
    assert!(dt > 0.0, "sampling period must be positive, got dt = {dt}");
    assert!(
        lb_hz >= 0.0,
        "line broadening must be non-negative, got lb_hz = {lb_hz}"
    );
    if lb_hz == 0.0 {
        return;
    }
    let alpha = PI * lb_hz;
    for (k, sample) in fid.iter_mut().enumerate() {
        let t = k as f64 * dt;
        let w = (-alpha * t).exp();
        *sample *= w;
    }
}

/// Zero-fill an FID by an integer factor.
///
/// Returns a new FID of length `fid.len() * factor`. The first `fid.len()`
/// samples are copied from the input; the tail is zero. FFT'ing the result
/// produces a spectrum with `factor` times as many frequency bins covering
/// the same ±Nyquist window — i.e. the spectrum is interpolated.
///
/// `factor == 1` returns the input unchanged (useful when the factor is a
/// config knob that may or may not be set).
///
/// # Panics
///
/// Panics if `factor == 0` or if the resulting length overflows `usize`.
pub fn zero_fill(fid: Vec<Complex<f64>>, factor: usize) -> Vec<Complex<f64>> {
    assert!(factor >= 1, "zero-fill factor must be >= 1, got {factor}");
    if factor == 1 {
        return fid;
    }
    let new_len = fid
        .len()
        .checked_mul(factor)
        .expect("zero-fill target length overflows usize");
    let mut padded = Vec::with_capacity(new_len);
    padded.extend(fid);
    padded.resize(new_len, Complex::new(0.0, 0.0));
    padded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discrete_spectrum::DiscreteSpectrum;

    // ---- apodize_exponential ----

    #[test]
    fn apodize_zero_lb_is_noop() {
        let orig: Vec<Complex<f64>> = (0..8).map(|k| Complex::new(k as f64, 0.0)).collect();
        let mut fid = orig.clone();
        apodize_exponential(&mut fid, 1e-4, 0.0);
        for (a, b) in fid.iter().zip(orig.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    #[test]
    fn apodize_first_sample_untouched() {
        // At t = 0 the envelope is exp(0) = 1, so sample 0 is unchanged.
        let mut fid = [Complex::new(1.0, 2.0); 16];
        apodize_exponential(&mut fid, 1.0 / 20_000.0, 1.0);
        assert!((fid[0] - Complex::new(1.0, 2.0)).norm() < 1e-12);
    }

    #[test]
    fn apodize_scales_exactly_by_expected_exponential() {
        // Start from all-ones; after apodization sample k should equal
        // exp(-π · lb · k · dt) exactly.
        let n = 64;
        let dt: f64 = 1.0 / 10_000.0;
        let lb_hz: f64 = 5.0;
        let mut fid = vec![Complex::new(1.0, 0.0); n];
        apodize_exponential(&mut fid, dt, lb_hz);
        for (k, sample) in fid.iter().enumerate() {
            let t = k as f64 * dt;
            let expected = (-PI * lb_hz * t).exp();
            assert!(
                (sample.re - expected).abs() < 1e-12,
                "sample {k}: got {}, expected {expected}",
                sample.re,
            );
            assert!(sample.im.abs() < 1e-12);
        }
    }

    #[test]
    fn apodize_monotonically_attenuates_amplitude() {
        // For any positive lb, |fid[k]| should be a non-increasing sequence
        // (strictly decreasing after the first sample).
        let n = 1000;
        let dt: f64 = 1e-4;
        let mut fid = vec![Complex::new(1.0, 0.0); n];
        apodize_exponential(&mut fid, dt, 3.0);
        for pair in fid.windows(2) {
            assert!(pair[0].norm() >= pair[1].norm());
        }
        // Last sample is meaningfully attenuated. With k = 999, dt = 1e-4,
        // lb = 3 Hz: exp(-π · 3 · 999 · 1e-4) = exp(-0.9414) ≈ 0.3901.
        assert!(fid[n - 1].norm() < 0.45);
        assert!(fid[n - 1].norm() > 0.35);
    }

    #[test]
    #[should_panic(expected = "line broadening must be non-negative")]
    fn apodize_rejects_negative_lb() {
        let mut fid = [Complex::new(1.0, 0.0); 4];
        apodize_exponential(&mut fid, 1e-4, -0.5);
    }

    #[test]
    #[should_panic(expected = "sampling period must be positive")]
    fn apodize_rejects_zero_dt() {
        let mut fid = [Complex::new(1.0, 0.0); 4];
        apodize_exponential(&mut fid, 0.0, 1.0);
    }

    // ---- zero_fill ----

    #[test]
    fn zero_fill_factor_one_is_identity() {
        let fid: Vec<Complex<f64>> = (0..10)
            .map(|k| Complex::new(k as f64, -(k as f64)))
            .collect();
        let padded = zero_fill(fid.clone(), 1);
        assert_eq!(padded, fid);
    }

    #[test]
    fn zero_fill_preserves_head_and_zeros_tail() {
        let fid = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        ];
        let padded = zero_fill(fid.clone(), 4);
        assert_eq!(padded.len(), 12);
        for (i, z) in padded.iter().enumerate() {
            if i < fid.len() {
                assert_eq!(*z, fid[i]);
            } else {
                assert_eq!(*z, Complex::new(0.0, 0.0));
            }
        }
    }

    #[test]
    #[should_panic(expected = "zero-fill factor must be >= 1")]
    fn zero_fill_rejects_zero_factor() {
        let fid = vec![Complex::new(1.0, 0.0); 4];
        let _ = zero_fill(fid, 0);
    }

    #[test]
    fn zero_fill_shrinks_bin_width_by_factor() {
        // N = 100, dt = 1e-4 → Δf = 100 Hz. Zero-fill 4x → Δf = 25 Hz.
        let n = 100;
        let dt: f64 = 1e-4;
        let fid = vec![Complex::new(1.0, 0.0); n];
        let spec_raw = DiscreteSpectrum::from_fid(&fid, dt);
        let df_raw = spec_raw.freq_resolution_hz();

        let padded = zero_fill(fid, 4);
        let spec_padded = DiscreteSpectrum::from_fid(&padded, dt);
        let df_padded = spec_padded.freq_resolution_hz();

        assert!((df_raw - 100.0).abs() < 1e-6, "raw df = {df_raw} Hz");
        assert!(
            (df_padded - 25.0).abs() < 1e-6,
            "padded df = {df_padded} Hz",
        );
    }

    // ---- End-to-end pipeline sanity check ----

    #[test]
    fn pipeline_locates_single_tone_and_suppresses_sinc_tails() {
        // Single complex tone at +100 Hz. Apodize with LB = 3 Hz and take
        // N · dt = 0.82 s so that the FID decays to ~4e-4 of its initial
        // amplitude before truncation — i.e. we really do see a Lorentzian,
        // not a boxcar-sinc with a mild envelope on top. Zero-fill 4× for a
        // smooth display grid.
        let n = 8192;
        let dt: f64 = 1.0 / 10_000.0;
        let lb_hz: f64 = 3.0;
        let omega: f64 = 2.0 * PI * 100.0;
        let mut fid: Vec<Complex<f64>> = (0..n)
            .map(|k| {
                let t = k as f64 * dt;
                Complex::from_polar(1.0, omega * t)
            })
            .collect();
        apodize_exponential(&mut fid, dt, lb_hz);
        let padded = zero_fill(fid, 4);
        let spec = DiscreteSpectrum::from_fid(&padded, dt);
        let mags = spec.magnitude();

        let (peak_idx, peak_mag) = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let peak_freq = spec.frequencies_hz[peak_idx];
        assert!(
            (peak_freq - 100.0).abs() < 0.5,
            "peak at {peak_freq} Hz, expected ~100 Hz",
        );

        // Beyond ±50 Hz from the peak a Lorentzian with α = π·lb_hz = 3π
        // has magnitude ratio α / √(α² + (2π·Δf)²) ≈ 3/100 ≈ 3% at Δf = 50 Hz.
        // 5% threshold gives headroom for the ~4e-4 residual boxcar tail.
        let threshold = 0.05 * peak_mag;
        for (i, &mag) in mags.iter().enumerate() {
            if (spec.frequencies_hz[i] - 100.0).abs() > 50.0 {
                assert!(
                    mag < threshold,
                    "out-of-band magnitude at {} Hz: {} (peak = {})",
                    spec.frequencies_hz[i],
                    mag,
                    peak_mag,
                );
            }
        }
    }
}
