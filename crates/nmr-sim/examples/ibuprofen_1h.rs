//! Demo: simulate the 1H NMR spectrum of ibuprofen at 600 MHz using the
//! first-order (Pascal's-triangle) backend.
//!
//! Ibuprofen 1H structure sketch:
//!   (CH3)2-CH-CH2-C6H4-CH(CH3)-COOH
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example ibuprofen_1h
//! ```
//!
//! Writes `examples/outputs/spectrum.csv` (created if absent), so generated
//! files stay out of the repo root and a single gitignore line covers every
//! example.

use std::fs::create_dir_all;

use nmr_sim::{Coupling, Peak, PeakDef, Spectrum};

/// Centralised output directory shared by every example. Lives at the
/// workspace root because `cargo run --example` runs from there.
const OUTPUT_DIR: &str = "examples/outputs";

fn main() {
    let spectrometer_freq = 600.0_f64; // MHz
    let linewidth = 1.0_f64; // Hz — narrow to resolve splitting

    let definitions = [
        PeakDef {
            // (CH3)2 — 6H, doublet
            shift: 0.90,
            area: 6.0,
            linewidth,
            couplings: vec![Coupling {
                j_hz: 6.6,
                n_neighbors: 1,
            }],
        },
        PeakDef {
            // CH3 on α-carbon — 3H, doublet
            shift: 1.52,
            area: 3.0,
            linewidth,
            couplings: vec![Coupling {
                j_hz: 7.1,
                n_neighbors: 1,
            }],
        },
        PeakDef {
            // CH (isopropyl) — 1H, septet
            shift: 1.85,
            area: 1.0,
            linewidth,
            couplings: vec![Coupling {
                j_hz: 6.6,
                n_neighbors: 6,
            }],
        },
        PeakDef {
            // CH2 benzylic — 2H, doublet
            shift: 2.47,
            area: 2.0,
            linewidth,
            couplings: vec![Coupling {
                j_hz: 7.2,
                n_neighbors: 1,
            }],
        },
        PeakDef {
            // CH (α) — 1H, quartet
            shift: 3.72,
            area: 1.0,
            linewidth,
            couplings: vec![Coupling {
                j_hz: 7.1,
                n_neighbors: 3,
            }],
        },
        PeakDef {
            // aromatic — 2H, triplet
            shift: 7.12,
            area: 2.0,
            linewidth,
            couplings: vec![Coupling {
                j_hz: 8.0,
                n_neighbors: 2,
            }],
        },
        PeakDef {
            // aromatic — 2H, triplet
            shift: 7.24,
            area: 2.0,
            linewidth,
            couplings: vec![Coupling {
                j_hz: 8.0,
                n_neighbors: 2,
            }],
        },
    ];

    let peaks: Vec<Peak> = definitions
        .iter()
        .flat_map(|def| def.expand(spectrometer_freq))
        .collect();
    println!("Expanded to {} sub-peaks", peaks.len());

    let spectrum = Spectrum::new(peaks, 0.0, 10.0, spectrometer_freq);
    create_dir_all(OUTPUT_DIR).expect("failed to create output directory");
    let path = format!("{OUTPUT_DIR}/spectrum.csv");
    match spectrum.to_csv(&path) {
        Ok(()) => println!("Saved {path}"),
        Err(e) => eprintln!("Error: {e}"),
    }
}
