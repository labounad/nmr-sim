//! Core spin-system vocabulary for the quantum engine.
//!
//! This module defines the types that describe *what* we're simulating,
//! independent of the linear-algebra machinery in [`crate::operator`] that
//! actually builds matrices for them.
//!
//! - [`Isotope`]  — an NMR-active nucleus (1H, 13C, ...) with its physics constants.
//! - [`Spin`]     — a single nuclear site in a molecule: an isotope plus a chemical shift.
//! - [`SpinSystem`] — a collection of spins in a static magnetic field B₀.
//!
//! # Units
//!
//! We stick to SI internally. Chemical shifts are stored in ppm (the conventional
//! spectroscopic unit; it's dimensionless). Gyromagnetic ratios are in rad·s⁻¹·T⁻¹.
//! Field strength is in Tesla. Larmor frequencies are derived on demand.
//!
//! # Spin quantum number convention
//!
//! We store spin as `two_i: u32` — i.e., 2I as an integer. Real nuclei always
//! have I ∈ {0, 1/2, 1, 3/2, 2, 5/2, 3, 7/2, ...}, so 2I is always a nonnegative
//! integer. Using an integer avoids f64-comparison headaches and makes the
//! Hilbert-space dimension (2I + 1) an exact `u32`.

/// An NMR-active nuclear isotope.
///
/// New variants can be added as needed; keep in sync with the trait-like
/// methods below which must return the right constants for each variant.
///
/// # Sign of γ
///
/// Gyromagnetic ratios can be positive *or negative*. A negative γ (notably
/// for 15N) means the nuclear magnetic moment is antiparallel to the spin
/// angular momentum, which flips the sign of the Larmor precession.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Isotope {
    /// Proton, I = 1/2, γ > 0. The workhorse of solution NMR.
    H1,
    /// Deuterium, I = 1 (!), γ > 0. Useful as a solvent lock and for 2D ssNMR.
    H2,
    /// Carbon-13, I = 1/2, γ > 0. Low natural abundance (~1%).
    C13,
    /// Nitrogen-15, I = 1/2, γ < 0. Very low natural abundance (~0.4%).
    N15,
    /// Fluorine-19, I = 1/2, γ > 0. 100% natural abundance, highly sensitive.
    F19,
    /// Phosphorus-31, I = 1/2, γ > 0. 100% natural abundance.
    P31,
}

// ---------- Nuclear constants ----------
//
// Values are from the IAEA / IUPAC recommended tables. If anyone wants more
// sig figs later it's a one-line change. γ is in rad·s⁻¹·T⁻¹.

const GAMMA_H1: f64 = 2.675_221_874_4e8;
const GAMMA_H2: f64 = 4.106_627_91e7;
const GAMMA_C13: f64 = 6.728_284e7;
const GAMMA_N15: f64 = -2.712_618_04e7; // negative!
const GAMMA_F19: f64 = 2.518_148e8;
const GAMMA_P31: f64 = 1.083_94e8;

impl Isotope {
    /// Twice the spin quantum number (2I), as an exact integer.
    ///
    /// Hilbert-space dimension of a single spin is `two_i() + 1`.
    pub const fn two_i(self) -> u32 {
        match self {
            Isotope::H1 | Isotope::C13 | Isotope::N15 | Isotope::F19 | Isotope::P31 => 1, // I = 1/2
            Isotope::H2 => 2,                                                             // I = 1
        }
    }

    /// Hilbert-space dimension of this single nucleus (2I + 1).
    pub const fn multiplicity(self) -> u32 {
        self.two_i() + 1
    }

    /// Gyromagnetic ratio γ, in rad·s⁻¹·T⁻¹. May be negative.
    pub const fn gamma(self) -> f64 {
        match self {
            Isotope::H1 => GAMMA_H1,
            Isotope::H2 => GAMMA_H2,
            Isotope::C13 => GAMMA_C13,
            Isotope::N15 => GAMMA_N15,
            Isotope::F19 => GAMMA_F19,
            Isotope::P31 => GAMMA_P31,
        }
    }

    /// Natural abundance as a mole fraction in [0, 1].
    pub const fn natural_abundance(self) -> f64 {
        match self {
            Isotope::H1 => 0.999_885,
            Isotope::H2 => 0.000_156,
            Isotope::C13 => 0.010_7,
            Isotope::N15 => 0.003_68,
            Isotope::F19 => 1.0,
            Isotope::P31 => 1.0,
        }
    }

    /// Larmor frequency ν₀ = -γ B₀ / (2π) at field strength `b0_tesla`, in Hz.
    ///
    /// This is the signed convention (so 15N at positive B₀ gives a positive
    /// Larmor frequency because γ is negative). Many textbooks drop the sign
    /// and quote |ν₀| — both conventions exist.
    pub fn larmor_hz(self, b0_tesla: f64) -> f64 {
        -self.gamma() * b0_tesla / (2.0 * std::f64::consts::PI)
    }
}

impl std::fmt::Display for Isotope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let disp = match self {
            Isotope::H1 => "1H",
            Isotope::H2 => "2H",
            Isotope::C13 => "13C",
            Isotope::N15 => "15N",
            Isotope::F19 => "19F",
            Isotope::P31 => "31P",
        };

        f.write_str(disp)
    }
}

/// A single nuclear site: *which* isotope, and *where* (chemical shift).
///
/// The chemical shift is the isotropic value δ, in ppm relative to the
/// appropriate reference (TMS for 1H/13C, etc.). CSA tensors, residual
/// dipolar couplings, etc. will hang off richer extensions later.
#[derive(Debug, Clone, PartialEq)]
pub struct Spin {
    pub isotope: Isotope,
    pub shift_ppm: f64,
    /// Optional human-readable label, e.g. "H_alpha", "C3". Used for display
    /// and debugging; never load-bearing for the physics.
    pub label: Option<String>,
}

impl Spin {
    /// Convenience constructor without a label.
    pub fn new(isotope: Isotope, shift_ppm: f64) -> Self {
        Self {
            isotope,
            shift_ppm,
            label: None,
        }
    }

    /// Convenience constructor with a label.
    pub fn labeled(isotope: Isotope, shift_ppm: f64, label: impl Into<String>) -> Self {
        Self {
            isotope,
            shift_ppm,
            label: Some(label.into()),
        }
    }

    /// The chemical-shift-only Larmor offset, in rad/s, assuming a rotating frame at the bare-isotope Larmor.
    /// Formula: Δω = −γ · B₀ · δ · 10⁻⁶.
    pub fn shift_angular(&self, b0_tesla: f64) -> f64 {
        -self.isotope.gamma() * b0_tesla * self.shift_ppm * 1e-6
    }

    /// The Larmor angular frequency ω₀ of this spin (in rad/s), accounting for the chemical shift.
    /// Formula: ω₀ = -γ · B₀ · (1 + δ · 10⁻⁶)
    pub fn larmor_angular(&self, b0_tesla: f64) -> f64 {
        -self.isotope.gamma() * b0_tesla * (1.0 + self.shift_ppm * 1e-6)
    }
}

impl std::fmt::Display for Spin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(label) = &self.label {
            write!(
                f,
                "{} ({} @ {:.3} ppm)",
                label, self.isotope, self.shift_ppm
            )
        } else {
            write!(f, "{} @ {:.3} ppm", self.isotope, self.shift_ppm)
        }
    }
}

/// A collection of spins in a static magnetic field.
///
/// The full Hilbert-space dimension is the product of the individual
/// multiplicities: dim = ∏ᵢ (2Iᵢ + 1). This grows exponentially with
/// the number of spins, which is why Liouville-space and reduced-basis
/// techniques exist — but for small systems the direct product is fine.
#[derive(Debug, Clone, PartialEq)]
pub struct SpinSystem {
    pub spins: Vec<Spin>,
    /// Static magnetic field, in Tesla.
    pub b0_tesla: f64,
}

impl SpinSystem {
    pub fn new(spins: Vec<Spin>, b0_tesla: f64) -> Self {
        Self { spins, b0_tesla }
    }

    /// Number of spins in the system.
    pub fn len(&self) -> usize {
        self.spins.len()
    }

    /// Whether the system has no spins.
    pub fn is_empty(&self) -> bool {
        self.spins.is_empty()
    }

    /// Full Hilbert-space dimension: ∏ᵢ (2Iᵢ + 1).
    ///
    /// Uses `u64` to stay exact for systems up to ~30 spin-1/2's, well past
    /// where we can actually compute in dense form.
    pub fn dim(&self) -> u64 {
        self.spins
            .iter()
            .map(|s| s.isotope.multiplicity() as u64)
            .product()
    }

    pub fn larmor_angulars(&self) -> Vec<f64> {
        self.spins
            .iter()
            .map(|spin| spin.larmor_angular(self.b0_tesla))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spin_half_multiplicity() {
        assert_eq!(Isotope::H1.multiplicity(), 2);
        assert_eq!(Isotope::C13.multiplicity(), 2);
        assert_eq!(Isotope::F19.multiplicity(), 2);
    }

    #[test]
    fn deuterium_is_spin_one() {
        assert_eq!(Isotope::H2.two_i(), 2);
        assert_eq!(Isotope::H2.multiplicity(), 3);
    }

    #[test]
    fn n15_gamma_is_negative() {
        assert!(Isotope::N15.gamma() < 0.0);
    }

    #[test]
    fn isotope_display_is_spectroscopic() {
        assert_eq!(format!("{}", Isotope::H1), "1H");
        assert_eq!(format!("{}", Isotope::C13), "13C");
        assert_eq!(format!("{}", Isotope::N15), "15N");
    }

    #[test]
    fn larmor_of_1h_at_14_1_t_is_roughly_600_mhz() {
        // 14.0954 T is the canonical "600 MHz" spectrometer field for 1H.
        let nu = Isotope::H1.larmor_hz(14.0954);
        // Sign convention gives a negative ν here (γ > 0, ν = -γB/2π),
        // so we compare absolute values.
        let mhz = nu.abs() / 1e6;
        assert!(
            (mhz - 600.0).abs() < 1.0,
            "expected ~600 MHz, got {mhz:.3} MHz",
        );
    }

    #[test]
    fn spin_system_dim_is_product_of_multiplicities() {
        // Two 1H's and one 2H: 2 * 2 * 3 = 12
        let sys = SpinSystem::new(
            vec![
                Spin::new(Isotope::H1, 1.0),
                Spin::new(Isotope::H1, 2.0),
                Spin::new(Isotope::H2, 3.0),
            ],
            14.0954,
        );
        assert_eq!(sys.len(), 3);
        assert_eq!(sys.dim(), 12);
    }

    #[test]
    fn empty_spin_system_has_dim_one() {
        // Vacuously, the dimension of a zero-spin Hilbert space is 1 (the
        // empty product). Useful base case for Kronecker-product code.
        let sys = SpinSystem::new(vec![], 14.0954);
        assert!(sys.is_empty());
        assert_eq!(sys.dim(), 1);
    }

    #[test]
    fn larmor_angular_at_zero_shift_matches_bare_isotope() {
        let spin = Spin::new(Isotope::H1, 0.0);
        let b0 = 14.0954;
        let expected_rad_per_s = Isotope::H1.larmor_hz(b0) * 2.0 * std::f64::consts::PI;
        assert!((spin.larmor_angular(b0) - expected_rad_per_s).abs() < 1e-3);
    }

    #[test]
    fn larmor_angular_shifts_linearly_with_ppm() {
        let b0 = 14.0954;
        let s_zero = Spin::new(Isotope::H1, 0.0);
        let s_ten = Spin::new(Isotope::H1, 10.0);
        let delta_rad_per_s = (s_ten.larmor_angular(b0) - s_zero.larmor_angular(b0)).abs();
        let expected = 6000.0 * 2.0 * std::f64::consts::PI; // (10 ppm or 6000 Hz in rad/s)
        assert!(
            (delta_rad_per_s - expected).abs() / expected < 1e-3,
            "got {delta_rad_per_s}, expected {expected}"
        );
    }

    #[test]
    fn spin_display_without_label() {
        let s = Spin::new(Isotope::H1, 1.23);
        assert_eq!(format!("{}", s), "1H @ 1.230 ppm");
    }

    #[test]
    fn spin_display_with_label() {
        let s = Spin::labeled(Isotope::C13, 128.4, "C_aromatic");
        assert_eq!(format!("{}", s), "C_aromatic (13C @ 128.400 ppm)");
    }

    #[test]
    fn larmor_angulars_length_matches_spin_count() {
        let sys = SpinSystem::new(
            vec![
                Spin::new(Isotope::H1, 0.0),
                Spin::new(Isotope::H1, 2.0),
                Spin::new(Isotope::C13, 50.0),
            ],
            14.0954,
        );
        assert_eq!(sys.larmor_angulars().len(), 3);
    }

    #[test]
    fn larmor_angulars_agree_with_per_spin_method() {
        let b0 = 14.0954;
        let spins = vec![Spin::new(Isotope::H1, 1.0), Spin::new(Isotope::C13, 77.0)];
        let sys = SpinSystem::new(spins.clone(), b0);
        let omegas = sys.larmor_angulars();

        for (i, spin) in spins.iter().enumerate() {
            assert!((omegas[i] - spin.larmor_angular(b0)).abs() < 1e-9);
        }
    }

    #[test]
    fn larmor_decomposes_into_bare_plus_shift() {
        let b0 = 14.0954;
        let s = Spin::new(Isotope::H1, 7.26); // chloroform proton, roughly
        let bare = -Isotope::H1.gamma() * b0; // rotating-frame reference
        let total = s.larmor_angular(b0);
        let offset = s.shift_angular(b0);
        assert!((total - (bare + offset)).abs() < 1e-3);
    }

    #[test]
    fn shift_angular_is_ppm_scale() {
        // 1 ppm at 14.0954 T should give ~3770 rad/s (600 Hz × 2π).
        let s = Spin::new(Isotope::H1, 1.0);
        let omega = s.shift_angular(14.0954).abs();
        assert!((omega - 2.0 * std::f64::consts::PI * 600.0).abs() < 5.0);
    }
}
