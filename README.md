# nmr-sim

> Fast, open-source spin-dynamics simulation for NMR — written in Rust.

**Status: 0.1.0 (bootstrap).** Public, early, and moving fast. The long-term
goal is a full Liouville-space simulator + Bruker/Varian I/O + spectrum/parameter
database + a Bayesian pipeline that updates an ML model for small-molecule
NMR spectrum → structure prediction.

The currently-shipping code is a first-order multiplet simulator (Pascal's-triangle
J-coupling expansion, Lorentzian lineshapes) — a useful baseline for weakly-coupled
1H spectra of small organics, and a test oracle for the forthcoming quantum engine.

If that sounds like a lot of words, see [`docs/architecture.md`](docs/architecture.md)
for the full plan.

---

## Quick start

```sh
# Requires stable Rust (edition 2021, MSRV 1.75).
cargo run --release --example ibuprofen_1h
# → writes spectrum.csv in the current directory

cargo test --workspace       # run the tests
cargo doc --workspace --open # browse the API docs
```

To use `nmr-sim` from another Rust project (once published):

```toml
[dependencies]
nmr-sim = "0.1"
```

Or as a git dependency today:

```toml
[dependencies]
nmr-sim = { git = "https://github.com/ORG/nmr-sim" }   # TODO: real URL
```

## Repository layout

```
nmr_sim/
├── Cargo.toml                     # workspace manifest
├── crates/
│   └── nmr-sim/                   # core simulation library
│       ├── src/
│       │   ├── lib.rs
│       │   ├── coupling.rs        # J-coupling expansion
│       │   ├── peak.rs            # Peak type + Lorentzian intensity
│       │   ├── spectrum.rs        # Spectrum aggregator + CSV output
│       │   └── template.rs        # Lorentzian interpolation template
│       └── examples/
│           └── ibuprofen_1h.rs    # runnable demo
├── docs/
│   ├── architecture.md            # design + roadmap (read this!)
│   ├── lorentzian_template_guide.html
│   └── doc_figures/               # figures referenced by the guide
└── scripts/                       # Python helpers (plotting, accuracy tests)
```

Upcoming crates (see architecture doc): `nmr-core`, `nmr-io`, `nmr-analysis`,
`nmr-cli`. The current `nmr-sim` crate will narrow to the simulation engine
as those split out.

## Roadmap

- **Phase 0 — Bootstrap (here).** Workspace, CI, license, first-order engine, docs.
- **Phase 1 — Quantum engine.** Density-matrix / Liouville-space propagator for
  arbitrary spin-1/2 systems; strong coupling; basic pulse sequences.
- **Phase 2 — Relaxation & exchange.** Redfield / Lindblad superoperators;
  chemical exchange; (stretch) MAS solid-state averaging.
- **Phase 3 — I/O.** Bruker and Varian raw-data readers; JCAMP-DX; nmrML.
- **Phase 4 — Platform.** Spectrum / parameter database; GUI for data ingest
  and exp-vs-sim comparison.
- **Phase 5 — ML.** Bayesian inference loop; spectrum → structure model for
  small molecules.

The roadmap is intentionally ambitious. Expect the phase numbers to blur.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). We welcome issues, PRs, and pre-PR
design discussions.

## License

Dual-licensed under either of

- Apache License, Version 2.0 — [`LICENSE-APACHE`](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>
- MIT license — [`LICENSE-MIT`](LICENSE-MIT) or
  <https://opensource.org/licenses/MIT>

at your option. This matches the convention in the Rust ecosystem.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual-licensed as above, without any additional terms or
conditions.
