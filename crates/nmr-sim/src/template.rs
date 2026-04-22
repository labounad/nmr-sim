// Lorentzian interpolation template
//
// The standard Lorentzian is L(x) = 1 / (x^2 + 1).
//
// Under the substitution x = tan(θ), this becomes L = cos²(θ).
// cos² is smooth and slowly varying, so linear interpolation
// in uniform θ-space is highly accurate — even where the original
// Lorentzian is sharp.
//
// Any specific peak's Lorentzian is recovered by:
//   x = (freq_hz - center_hz) / half_linewidth
//   value = peak_height * template.evaluate(x)

use crate::peak::Peak;

#[derive(Debug)]
pub struct LorentzianTemplate {
    cos2_values: Vec<f64>, // pre-computed cos²(θ) at each grid point
    theta_max: f64,        // cutoff angle — beyond this we return 0
    num_points: usize,     // number of sample points
    x_max: f64,            // tan(theta_max) — cutoff in x-space
}

impl LorentzianTemplate {
    /// Create a new template with `num_points` sample points.
    ///
    /// `theta_max` controls how far into the tails we go.
    /// A value of 1.5 (just under π/2 ≈ 1.5708) covers most
    /// of the Lorentzian; the remaining tail contributes negligibly.
    pub fn new(num_points: usize, theta_max: f64) -> LorentzianTemplate {
        let step = 2.0 * theta_max / (num_points as f64 - 1.0);

        let cos2_values: Vec<f64> = (0..num_points)
            .map(|i| {
                let theta = -theta_max + i as f64 * step;
                theta.cos().powi(2)
            })
            .collect();

        let x_max = theta_max.tan();

        LorentzianTemplate {
            cos2_values,
            theta_max,
            num_points,
            x_max,
        }
    }

    /// Evaluate the standard Lorentzian at x via interpolation in θ-space.
    ///
    /// x is the standardized frequency: (freq_hz - center_hz) / half_linewidth.
    /// Returns 0.0 if x is beyond the tail cutoff.
    pub fn evaluate(&self, x: f64) -> f64 {
        // Convert x to θ
        let theta = x.atan();

        // Beyond cutoff — negligible contribution
        if theta.abs() >= self.theta_max {
            return 0.0;
        }

        // Map θ from [-theta_max, theta_max] to index space [0, num_points - 1]
        let t = (theta + self.theta_max) / (2.0 * self.theta_max) * (self.num_points as f64 - 1.0);

        // Linear interpolation between the two nearest grid points
        let i = t.floor() as usize;
        if i >= self.num_points - 1 {
            return self.cos2_values[self.num_points - 1];
        }
        let frac = t - i as f64;
        self.cos2_values[i] * (1.0 - frac) + self.cos2_values[i + 1] * frac
    }

    /// Evaluate a specific peak's Lorentzian at freq_hz using the template.
    ///
    /// Converts freq_hz to standardized x, interpolates, and scales
    /// by the peak's intensity.
    pub fn evaluate_peak(&self, peak: &Peak, freq_hz: f64, spectrometer_freq: f64) -> f64 {
        let shift_hz = peak.shift * spectrometer_freq;
        let half_lw = peak.linewidth / 2.0;
        let x = (freq_hz - shift_hz) / half_lw;

        // Skip if beyond cutoff — no need to interpolate
        if x.abs() > self.x_max {
            return 0.0;
        }

        peak.intensity() * self.evaluate(x)
    }
}
