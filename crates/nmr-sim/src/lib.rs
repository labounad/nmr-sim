//! # nmr-sim
//!
//! Open-source spin dynamics simulation for NMR. This crate is the simulation
//! core of a larger project — see the repository root's `README.md` and
//! `docs/architecture.md` for the full vision (spin-dynamics library → Bruker /
//! Varian I/O → experimental/simulated spectrum DB → Bayesian ML loop →
//! spectrum-to-structure model).
//!
//! ## Current state (v0.1.0 — bootstrap)
//!
//! This release ships a **first-order multiplet simulator**: Pascal's-triangle
//! J-coupling expansion, Lorentzian lineshapes, summed to produce a 1D 1H NMR
//! spectrum. The math is exact for weakly-coupled, first-order systems (e.g.,
//! most routine 1H spectra of small organic molecules at high field).
//!
//! The simulator is implemented with a deliberate performance trick: the
//! standard Lorentzian `L(x) = 1 / (x^2 + 1)` becomes `cos²(θ)` under the
//! substitution `x = tan(θ)`, which is smooth enough to interpolate linearly
//! in θ-space with very high accuracy. See [`template`] for details.
//!
//! ## Where we're going
//!
//! The next major milestone is a density-matrix / Liouville-space propagator
//! capable of simulating arbitrary spin-1/2 (and, eventually, higher-spin)
//! systems with strong coupling, relaxation (Redfield / Lindblad), and chemical
//! exchange. The current first-order code will be preserved as a fast path and
//! a pedagogical baseline under a `backend::first_order` module (not yet split
//! out). Don't write code that assumes the first-order engine is the only
//! engine.
//!
//! ## Quick start
//!
//! ```ignore
//! use nmr_sim::{Coupling, PeakDef, Peak, Spectrum};
//!
//! let spectrometer_freq = 600.0_f64; // MHz
//! let linewidth = 1.0_f64;           // Hz
//!
//! let defs = vec![
//!     PeakDef { shift: 0.90, area: 6.0, linewidth,
//!               couplings: vec![Coupling { j_hz: 6.6, n_neighbors: 1 }] },
//!     // …
//! ];
//!
//! let peaks: Vec<Peak> = defs.iter().flat_map(|d| d.expand(spectrometer_freq)).collect();
//! let spectrum = Spectrum::new(peaks, 0.0, 10.0, spectrometer_freq);
//! spectrum.to_csv("spectrum.csv").unwrap();
//! ```
//!
//! See `examples/ibuprofen_1h.rs` for a complete, runnable demonstration.

#![doc(html_root_url = "https://docs.rs/nmr-sim")]

pub mod coupling;
pub mod peak;
pub mod spectrum;
pub mod template;

pub use coupling::{expand_peak, pascal_row, Coupling};
pub use peak::Peak;
pub use spectrum::Spectrum;
pub use template::LorentzianTemplate;

/// A chemically-distinct NMR signal before J-coupling is expanded into
/// sub-peaks.
///
/// `PeakDef` groups the physical parameters that describe one site:
///
/// - `shift`: chemical shift in ppm
/// - `area`: relative integrated intensity (proportional to the number of
///   contributing protons)
/// - `linewidth`: full-width at half maximum of the Lorentzian, in Hz
/// - `couplings`: J-couplings to sets of equivalent neighbors
///
/// Call [`PeakDef::expand`] to materialize this definition into the concrete
/// [`Peak`]s that a [`Spectrum`] consumes.
#[derive(Debug, Clone)]
pub struct PeakDef {
    /// Chemical shift, in ppm.
    pub shift: f64,
    /// Relative area (typically proportional to the number of protons).
    pub area: f64,
    /// Lorentzian full-width at half maximum, in Hz.
    pub linewidth: f64,
    /// J-couplings to neighboring spin groups.
    pub couplings: Vec<Coupling>,
}

impl PeakDef {
    /// Expand this definition into concrete [`Peak`]s by applying each
    /// J-coupling as a multiplet with Pascal's-triangle intensities.
    ///
    /// # Panics
    ///
    /// Panics if `linewidth` is not strictly positive. Construct `PeakDef`
    /// with a positive `linewidth` to avoid this; in a future revision the
    /// validation will be lifted into `PeakDef::new` returning `Result`.
    #[must_use]
    pub fn expand(&self, spectrometer_freq: f64) -> Vec<Peak> {
        expand_peak(self.shift, self.area, &self.couplings, spectrometer_freq)
            .into_iter()
            .map(|(shift, area)| {
                Peak::new(shift, area, self.linewidth)
                    .expect("PeakDef.linewidth must be strictly positive")
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_def_expand_preserves_total_area() {
        let def = PeakDef {
            shift: 1.5,
            area: 6.0,
            linewidth: 1.0,
            couplings: vec![
                Coupling {
                    j_hz: 7.0,
                    n_neighbors: 3,
                },
                Coupling {
                    j_hz: 5.0,
                    n_neighbors: 2,
                },
            ],
        };
        let peaks = def.expand(600.0);
        let total: f64 = peaks.iter().map(|p| p.area).sum();
        assert!((total - 6.0).abs() < 1e-10);
    }

    #[test]
    fn peak_def_without_couplings_yields_one_peak() {
        let def = PeakDef {
            shift: 2.0,
            area: 3.0,
            linewidth: 0.5,
            couplings: vec![],
        };
        let peaks = def.expand(500.0);
        assert_eq!(peaks.len(), 1);
        assert_eq!(peaks[0].shift, 2.0);
        assert_eq!(peaks[0].area, 3.0);
        assert_eq!(peaks[0].linewidth, 0.5);
    }
}
