use std::fs::File;
use std::io::Write;

use crate::peak::Peak;
use crate::template::LorentzianTemplate;

// Sensible defaults for template configuration
const DEFAULT_TEMPLATE_POINTS: usize = 100;
const DEFAULT_THETA_MAX: f64 = 1.55; // PI/2 ~ 1.57
const DEFAULT_NUM_POINTS: usize = 200_000;

#[derive(Debug)]
pub struct Spectrum {
    pub peaks: Vec<Peak>,
    pub freq_start: f64,        // ppm
    pub freq_end: f64,          // ppm
    pub spectrometer_freq: f64, // MHz
    pub num_points: usize,
    pub template_points: usize,
    pub theta_max: f64,
}

impl Spectrum {
    /// Create a spectrum with default template settings.
    pub fn new(
        peaks: Vec<Peak>,
        freq_start: f64,
        freq_end: f64,
        spectrometer_freq: f64,
    ) -> Spectrum {
        Spectrum {
            peaks,
            freq_start,
            freq_end,
            spectrometer_freq,
            num_points: DEFAULT_NUM_POINTS,
            template_points: DEFAULT_TEMPLATE_POINTS,
            theta_max: DEFAULT_THETA_MAX,
        }
    }

    /// Override the number of output points.
    pub fn with_num_points(mut self, n: usize) -> Spectrum {
        self.num_points = n;
        self
    }

    /// Override the template interpolation settings.
    pub fn with_template(mut self, points: usize, theta_max: f64) -> Spectrum {
        self.template_points = points;
        self.theta_max = theta_max;
        self
    }

    /// Compute the spectrum using the interpolation template.
    pub fn compute(&self) -> Vec<(f64, f64)> {
        let template = LorentzianTemplate::new(self.template_points, self.theta_max);

        let start_hz = self.freq_start * self.spectrometer_freq;
        let end_hz = self.freq_end * self.spectrometer_freq;
        let step_size = (end_hz - start_hz) / (self.num_points as f64 - 1.0);

        let freq_grid: Vec<f64> = (0..self.num_points)
            .map(|i| start_hz + i as f64 * step_size)
            .collect();

        freq_grid
            .iter()
            .map(|&freq| {
                let intensity: f64 = self
                    .peaks
                    .iter()
                    .map(|peak| template.evaluate_peak(peak, freq, self.spectrometer_freq))
                    .sum();
                (freq, intensity)
            })
            .collect()
    }

    pub fn to_csv(&self, filename: &str) -> Result<(), String> {
        let data = self.compute();
        let mut file =
            File::create(filename).map_err(|e| format!("Could not create file: {}", e))?;
        writeln!(file, "chemical_shift_ppm,intensity")
            .map_err(|e| format!("Write error: {}", e))?;
        for (freq_hz, intensity) in &data {
            let ppm = freq_hz / self.spectrometer_freq;
            writeln!(file, "{},{}", ppm, intensity).map_err(|e| format!("Write error: {}", e))?;
        }
        Ok(())
    }
}
