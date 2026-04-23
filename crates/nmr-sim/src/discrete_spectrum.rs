//! Sampled frequency-domain spectra — the FFT output of an FID.
//!
//! This complements [`crate::spectrum::Spectrum`], which is the *parametric*
//! spectrum (a list of peaks evaluated analytically against a Lorentzian
//! template). A [`DiscreteSpectrum`], by contrast, is an array of complex
//! amplitudes on a uniform frequency grid — the direct output of
//! FFT'ing the time-domain FID returned by [`crate::fid::compute_fid`].
//!
//! Both types will coexist: parametric spectra remain the fastest path for
//! the first-order backend and for plotting from a peak list, while discrete
//! spectra are what the quantum engine naturally produces and what comparison
//! against experimental data will consume.
//!
//! # Conventions
//!
//! After [`DiscreteSpectrum::from_fid`]:
//!
//! - The frequency axis is centered (fftshifted), running from the most
//!   negative offset to the most positive in uniform
//!   [`freq_resolution_hz`](DiscreteSpectrum::freq_resolution_hz) steps.
//! - Amplitudes are the raw complex FFT values, also fftshifted. Callers
//!   choose their presentation: magnitude ([`DiscreteSpectrum::magnitude`])
//!   is simplest and phase-free; real/imag expose the absorption / dispersion
//!   split that a phase-corrected experimental spectrum would show.
//! - An unapodized FID produces a spectrum with sinc-shaped peaks; no
//!   windowing is applied here. Apodization arrives in a future milestone
//!   along with relaxation / T₂ models.
//!
//! # Converting Hz ↔ ppm
//!
//! Hz → ppm requires a choice of *reference isotope*: chemical shift is the
//! offset in rotating-frame angular frequency divided by the reference
//! Larmor. Our FID lives in the rotating frame of each isotope (by
//! construction in `hamiltonian::ZeemanH`), so for a homonuclear experiment
//! the conversion is straightforward. For heteronuclear spectra the user
//! has to pick which channel to plot; see [`DiscreteSpectrum::frequencies_ppm`].

use crate::spin::Isotope;
use num_complex::Complex;
use rustfft::FftPlanner;

/// A sampled frequency-domain NMR spectrum, produced by FFT of an FID.
///
/// Stores the frequency axis (in Hz) and the complex FFT amplitudes,
/// both fftshifted so that DC is in the middle and negative offsets come
/// first.
#[derive(Debug, Clone)]
pub struct DiscreteSpectrum {
    /// Frequency of each bin, in Hz. Strictly ascending.
    pub frequencies_hz: Vec<f64>,
    /// Complex amplitude at each bin, in the same order.
    pub amplitudes: Vec<Complex<f64>>,
}

impl DiscreteSpectrum {
    /// FFT a time-domain FID into a frequency-domain spectrum.
    ///
    /// `fid[k]` is assumed to be the signal at time `t_k = k · dt`. The
    /// FFT runs in-place on a clone of the FID; the returned spectrum is
    /// fftshifted (negative frequencies first, DC in the middle).
    ///
    /// # Sign convention
    ///
    /// `rustfft`'s forward transform is `S[m] = Σₖ s[k] · exp(−2πi km/N)`,
    /// so a signal `s(t) = exp(+iΩt)` maps to a peak at the positive
    /// frequency `Ω / 2π`. Combined with our choice of Ô = M⁻ in
    /// [`crate::fid::compute_fid`] — which makes each spin oscillate as
    /// `exp(−iΔωt)` — this places a chemical-shift-offset peak at `−Δω/2π`
    /// in Hz. For a 1H (γ > 0) at positive ppm that is a positive Hz
    /// offset, and therefore a positive ppm after the isotope-aware
    /// conversion in [`Self::frequencies_ppm`].
    ///
    /// # Panics
    ///
    /// Panics if `fid` is empty.
    ///
    /// # Complexity
    ///
    /// O(N log N). For typical NMR sizes (N = 8k–64k) this is sub-millisecond;
    /// the dominant cost of the full pipeline stays in propagation.
    pub fn from_fid(fid: &[Complex<f64>], dt: f64) -> Self {
        assert!(!fid.is_empty(), "cannot FFT an empty FID");
        assert!(dt > 0.0, "sampling period must be positive, got {dt}");

        let n = fid.len();
        let mut buffer = fid.to_vec();

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buffer);

        // Build the "natural" (unshifted) frequency axis following the
        // numpy.fft.fftfreq convention, which handles even/odd N uniformly:
        //
        //   bin m ↔ f = m · Δf                for m in [0, split)
        //   bin m ↔ f = (m − N) · Δf          for m in [split, N)
        //
        //   where split = ⌈N / 2⌉ = (N + 1) / 2 (integer div).
        //
        // For even N, `split = N/2` and bin N/2 is Nyquist, assigned to the
        // negative side by convention. For odd N, `split` is the first bin
        // past DC + the ⌊N/2⌋ positive bins.
        let df = 1.0 / (n as f64 * dt);
        let split = n.div_ceil(2);
        let mut freqs: Vec<f64> = (0..n)
            .map(|m| {
                if m < split {
                    m as f64 * df
                } else {
                    (m as f64 - n as f64) * df
                }
            })
            .collect();

        // fftshift: rotate so the axis is monotonically ascending (negatives
        // first, DC in the middle for even N, at index ⌊N/2⌋ for odd N).
        freqs.rotate_left(split);
        buffer.rotate_left(split);

        Self {
            frequencies_hz: freqs,
            amplitudes: buffer,
        }
    }

    /// Number of points in the spectrum.
    pub fn len(&self) -> usize {
        self.amplitudes.len()
    }

    /// Whether the spectrum has no points. Always `false` for a spectrum
    /// constructed via [`Self::from_fid`] (that constructor rejects empty
    /// FIDs), but provided for lint cleanliness given we implement `len`.
    pub fn is_empty(&self) -> bool {
        self.amplitudes.is_empty()
    }

    /// Frequency step between adjacent bins, in Hz.
    ///
    /// Equal to `1 / (N · dt)` by construction. Returns 0 for a
    /// single-point spectrum.
    pub fn freq_resolution_hz(&self) -> f64 {
        if self.frequencies_hz.len() < 2 {
            0.0
        } else {
            self.frequencies_hz[1] - self.frequencies_hz[0]
        }
    }

    /// Magnitude spectrum: `|S[m]|` at each bin.
    ///
    /// Simplest, phase-free presentation. Peaks are always positive but are
    /// broader than the corresponding absorption-mode peaks (√2× FWHM for a
    /// Lorentzian line).
    pub fn magnitude(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|z| z.norm()).collect()
    }

    /// Real part of each bin: `Re S[m]`.
    ///
    /// After zeroth-order phase correction this is the absorption-mode
    /// spectrum. We don't auto-phase; callers currently get whatever phase
    /// the raw FFT produces.
    pub fn real(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|z| z.re).collect()
    }

    /// Imaginary part of each bin: `Im S[m]`. After phase correction, the
    /// dispersion-mode spectrum.
    pub fn imag(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|z| z.im).collect()
    }

    /// Convert the frequency axis from Hz to ppm relative to a reference
    /// isotope at field `b0_tesla`.
    ///
    /// Derivation. Our Hamiltonian convention (see [`crate::hamiltonian`]) is
    /// Δω = −γ · B₀ · δ · 10⁻⁶, and with Ô = M⁻ a spin at chemical shift δ
    /// contributes a FID component `(γ/2) · exp(−iΔω·t)`. After FFT the
    /// peak lands at Hz frequency `f = −Δω / (2π)`. Inverting both relations:
    ///
    /// ```text
    /// Δω = −2π · f_Hz
    /// δ [ppm] = −Δω · 10⁶ / (γ · B₀)
    ///         = 2π · f_Hz · 10⁶ / (γ · B₀)
    /// ```
    ///
    /// For a 1H (γ > 0) this maps positive Hz → positive ppm, matching
    /// the conventional "downfield = positive" NMR display. For 15N
    /// (γ < 0) the mapping inverts the sign, as it should — that's the
    /// physical content of γ_{15N} < 0.
    ///
    /// # Panics
    ///
    /// Panics if `b0_tesla` is zero (ppm would be undefined).
    pub fn frequencies_ppm(&self, reference_isotope: Isotope, b0_tesla: f64) -> Vec<f64> {
        assert!(
            b0_tesla != 0.0,
            "b0_tesla must be nonzero to convert to ppm"
        );
        let gamma = reference_isotope.gamma();
        let hz_to_ppm = 2.0 * std::f64::consts::PI * 1e6 / (gamma * b0_tesla);
        self.frequencies_hz.iter().map(|f| f * hz_to_ppm).collect()
    }

    /// Write the absorption-mode spectrum to CSV, in ppm.
    ///
    /// Columns: `chemical_shift_ppm,intensity`. Sorted in the usual NMR
    /// display order: ppm descending (leftmost column is the most positive
    /// shift — i.e., the left edge of the plot), so the file is directly
    /// plottable without a reversing step.
    ///
    /// "Intensity" is the **real part** of the complex FFT — i.e. the
    /// absorption-mode spectrum. This matches what commercial NMR software
    /// (Mestrenova, Topspin, NMRPipe) displays after phase correction:
    /// symmetric Lorentzian lineshapes with fast-decaying tails and a
    /// baseline that genuinely returns to zero (positive and negative sinc
    /// side lobes cancel out).
    ///
    /// # Phase assumption
    ///
    /// This method assumes the spectrum is already phased — i.e. that the
    /// underlying FID is purely real at `t = 0`. The standard pipeline in
    /// this crate (thermal `I_x` state, `M⁻` observable, no pulse sequence)
    /// satisfies that by construction: `Tr[M⁻ · ρ₀]` is real positive, so
    /// `Re{FFT}` is the absorption-mode line and no phase correction is
    /// needed.
    ///
    /// If you've introduced an arbitrary initial phase (by applying pulses,
    /// starting from a non-standard state, etc.), the real part will be a
    /// mixture of absorption and dispersion and the lineshapes will look
    /// twisted. Phase-correct first, or fall back to [`Self::magnitude`] +
    /// your own CSV writer.
    pub fn to_csv_ppm(
        &self,
        filename: &str,
        reference_isotope: Isotope,
        b0_tesla: f64,
    ) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let ppms = self.frequencies_ppm(reference_isotope, b0_tesla);
        let intensities = self.real();
        let mut rows: Vec<(f64, f64)> = ppms.into_iter().zip(intensities).collect();
        // Descending ppm — downfield on the left, upfield on the right,
        // matching how spectrometers plot 1H spectra.
        rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut file = File::create(filename)?;
        writeln!(file, "chemical_shift_ppm,intensity")?;
        for (ppm, amp) in rows {
            writeln!(file, "{ppm},{amp}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fid::compute_fid;
    use crate::hamiltonian::ZeemanH;
    use crate::operator::total_m_minus;
    use crate::propagator::DiagonalPropagator;
    use crate::spin::{Isotope, Spin, SpinSystem};
    use crate::state::thermal_x_state;

    // ---- FFT axis layout ----

    #[test]
    fn frequency_axis_is_monotonic_and_centered() {
        // Build a 16-point spectrum from a trivial FID (all zeros). We only
        // care about the axis here.
        let fid = vec![Complex::new(0.0, 0.0); 16];
        let spec = DiscreteSpectrum::from_fid(&fid, 1e-5);
        // Monotonic ascending.
        for pair in spec.frequencies_hz.windows(2) {
            assert!(pair[1] > pair[0], "axis not strictly ascending: {pair:?}");
        }
        // Uniform spacing. Tolerance is in absolute Hz; df ≈ 6250 Hz, and
        // each bin's value comes from `m * df` or `(m - N) * df` which can
        // differ by a few ULPs (~1e-12 relative → ~1e-8 absolute) when N·dt
        // isn't a neat binary fraction. 1e-9 is tight enough to catch any
        // real spacing bug without tripping on roundoff.
        let df = spec.freq_resolution_hz();
        for pair in spec.frequencies_hz.windows(2) {
            assert!(((pair[1] - pair[0]) - df).abs() < 1e-9);
        }
        // Resolution matches 1/(N·dt).
        let expected_df = 1.0 / (16.0 * 1e-5);
        assert!((df - expected_df).abs() < 1e-9);
    }

    #[test]
    fn dc_fid_puts_all_intensity_at_zero_hz() {
        // A constant FID s[k] = c should have all spectral weight at f = 0.
        let c = Complex::new(1.0, 0.0);
        let n = 32;
        let fid = vec![c; n];
        let spec = DiscreteSpectrum::from_fid(&fid, 5e-6);
        let mags = spec.magnitude();
        // Find the zero-Hz bin.
        let dc_bin = spec
            .frequencies_hz
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap()
            .0;
        assert!(spec.frequencies_hz[dc_bin].abs() < 1e-9);
        // That bin should hold all the signal; everything else should be 0.
        for (m, &mag) in mags.iter().enumerate() {
            if m == dc_bin {
                assert!((mag - n as f64).abs() < 1e-9, "DC bin {m}: mag={mag}");
            } else {
                assert!(mag < 1e-9, "non-DC bin {m} leaked: mag={mag}");
            }
        }
    }

    // ---- End-to-end: single spin → peak at the right ppm ----

    #[test]
    fn single_spin_peak_lands_at_correct_ppm() {
        // Build the end-to-end pipeline for one 1H at δ = 5.0 ppm and check
        // the magnitude spectrum peaks near 5.0 ppm. This is the first
        // end-to-end validation of the quantum engine.
        let b0 = 14.0954;
        let delta_ppm = 5.0;
        let sys = SpinSystem::new(vec![Spin::new(Isotope::H1, delta_ppm)], b0);
        let h = ZeemanH::new(&sys);
        let dt = 1.0 / 20_000.0; // 20 kHz sampling — Nyquist safely covers ~±8 kHz = ±13 ppm at 600 MHz
        let n = 8192;
        let p = DiagonalPropagator::new(&h, dt);
        let rho0 = thermal_x_state(&sys);
        let obs = total_m_minus(&sys);
        let fid = compute_fid(&p, &rho0, &obs, n);
        let spec = DiscreteSpectrum::from_fid(&fid, dt);

        let ppms = spec.frequencies_ppm(Isotope::H1, b0);
        let mags = spec.magnitude();
        let peak_idx = mags
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let peak_ppm = ppms[peak_idx];
        // Frequency resolution at this (N, dt) corresponds to roughly
        //   Δf = 1/(N·dt) = 1/(8192 · 5e-5) ≈ 2.44 Hz ≈ 0.0041 ppm at 600 MHz
        // So we expect the peak to land within ~0.01 ppm of δ.
        assert!(
            (peak_ppm - delta_ppm).abs() < 0.05,
            "peak at {peak_ppm} ppm, expected {delta_ppm} ppm (Δf ≈ {} Hz)",
            spec.freq_resolution_hz(),
        );
    }

    #[test]
    fn two_spin_spectrum_has_two_peaks_at_correct_ppms() {
        // Two non-interacting 1H's at 1.5 and 7.3 ppm — confirm both peaks
        // appear at the right places. Without J-coupling, the spectrum
        // should be two clean lines of equal height.
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
        let dt = 1.0 / 20_000.0;
        let n = 8192;
        let p = DiagonalPropagator::new(&h, dt);
        let rho0 = thermal_x_state(&sys);
        let obs = total_m_minus(&sys);
        let fid = compute_fid(&p, &rho0, &obs, n);
        let spec = DiscreteSpectrum::from_fid(&fid, dt);

        let ppms = spec.frequencies_ppm(Isotope::H1, b0);
        let mags = spec.magnitude();

        // For each expected peak, find the max magnitude within ±0.1 ppm
        // and confirm the max is (a) close to the expected ppm and (b)
        // large compared to the off-peak noise.
        for &expected_ppm in &shifts {
            let (best_idx, best_mag) = mags
                .iter()
                .enumerate()
                .filter(|(i, _)| (ppms[*i] - expected_ppm).abs() < 0.1)
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap_or_else(|| panic!("no bin found near {expected_ppm} ppm"));
            assert!(
                (ppms[best_idx] - expected_ppm).abs() < 0.05,
                "peak for {expected_ppm} ppm landed at {} ppm",
                ppms[best_idx],
            );
            // Expected height: each spin contributes (γ/2)·N to the FFT
            // peak amplitude. We only check we're well above floor.
            assert!(
                *best_mag > 1e3,
                "peak at {expected_ppm} ppm has magnitude {best_mag} — suspiciously small",
            );
        }
    }

    #[test]
    fn heteronuclear_amplitudes_track_gamma_ratio() {
        // A single 13C and a single 1H should produce spectra whose total
        // spectral *energy* scales as |γ|, because `thermal_x_state` weights
        // each spin's contribution by its gyromagnetic ratio.
        //
        // We compare L2 norms (||spec||₂ = sqrt(Σ |S[m]|²)) rather than peak
        // heights. Rationale: peak-height comparisons are fragile because a
        // pure complex-exponential FID doesn't land on an integer FFT bin in
        // general, so its energy spreads across adjacent bins by a different
        // amount depending on the peak location — biasing the max magnitude
        // downward by a frequency-dependent factor of a few percent. The L2
        // norm, by Parseval's theorem, is *exactly* `sqrt(N)` times the
        // time-domain L2 norm of the FID and therefore independent of bin
        // placement and any aliasing wraparound. That is the clean,
        // location-free quantity to compare.
        let b0 = 14.0954;
        let n = 4096;
        // 50 kHz sampling keeps 13C@100ppm (~15 kHz offset) well inside
        // Nyquist. L2 norm would still be correct even under aliasing
        // (sum-of-squares doesn't care where the bin lands), but avoiding
        // aliasing is still good hygiene.
        let dt = 1.0 / 50_000.0;

        let sys_h = SpinSystem::new(vec![Spin::new(Isotope::H1, 5.0)], b0);
        let h_h = ZeemanH::new(&sys_h);
        let p_h = DiagonalPropagator::new(&h_h, dt);
        let rho_h = thermal_x_state(&sys_h);
        let obs_h = total_m_minus(&sys_h);
        let fid_h = compute_fid(&p_h, &rho_h, &obs_h, n);
        let spec_h = DiscreteSpectrum::from_fid(&fid_h, dt);
        let norm_h: f64 = spec_h.magnitude().iter().map(|m| m * m).sum::<f64>().sqrt();

        let sys_c = SpinSystem::new(vec![Spin::new(Isotope::C13, 100.0)], b0);
        let h_c = ZeemanH::new(&sys_c);
        let p_c = DiagonalPropagator::new(&h_c, dt);
        let rho_c = thermal_x_state(&sys_c);
        let obs_c = total_m_minus(&sys_c);
        let fid_c = compute_fid(&p_c, &rho_c, &obs_c, n);
        let spec_c = DiscreteSpectrum::from_fid(&fid_c, dt);
        let norm_c: f64 = spec_c.magnitude().iter().map(|m| m * m).sum::<f64>().sqrt();

        let expected = (Isotope::C13.gamma() / Isotope::H1.gamma()).abs();
        let actual = norm_c / norm_h;
        assert!(
            (actual - expected).abs() / expected < 1e-10,
            "γ ratio wrong: expected {expected}, got {actual}",
        );
    }
}
