use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Peak {
    pub shift: f64,     // ppm
    pub area: f64,      // relative
    pub linewidth: f64, // Hz
}

impl Peak {
    pub fn new(shift: f64, area: f64, linewidth: f64) -> Result<Peak, String> {
        if linewidth > 0.0 {
            Ok(Peak {
                shift,
                area,
                linewidth,
            })
        } else {
            Err(format!(
                "Linewidth {} is less than or equal to 0.",
                linewidth
            ))
        }
    }

    pub fn intensity(&self) -> f64 {
        2.0 * self.area / (PI * self.linewidth)
    }
}
