# Contributing to nmr-sim

Thanks for being here. This project is in its early days; every thoughtful
issue and PR genuinely moves it forward.

## The lay of the land

- **Design doc:** [`docs/architecture.md`](docs/architecture.md). Please skim
  it before proposing anything structural. If your idea is at odds with the
  doc, that's a fine conversation to have — just open an issue rather than a
  large PR.
- **Status:** pre-1.0. Public APIs will break. Don't depend on stability yet.

## Development workflow

### Prerequisites

- Stable Rust (MSRV 1.75+). Get it from <https://rustup.rs>.
- `rustfmt` and `clippy` components (rustup installs them by default).

### Common commands

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --all
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -Dwarnings
cargo doc --workspace --no-deps --open
cargo run --release --example ibuprofen_1h
```

CI runs `fmt --check`, `clippy -Dwarnings`, `test`, and `doc` on every push
and PR. If CI is green locally, it should be green in GitHub Actions.

## Scope of a good PR

- **Small.** Prefer 200-line PRs over 2000-line PRs. Split things up.
- **Tested.** For physics/math code, include a test with a known analytical
  answer (or a comparison against an established simulator).
- **Documented.** Public API items get doc comments. If you're adding a
  non-obvious algorithm, a short `//!` module header or block comment goes a
  long way — imagine a reader who knows NMR but not your specific trick.

## Scientific rigor

This is a simulation library. Numerical results matter. When you add or
change physics:

1. Describe the physical model in the PR (or link to a reference).
2. Add at least one test whose expected value comes from an independent
   source — an analytical result, a published table, or a cross-check
   against another package (Spinach, SIMPSON, NMRPipe, SpinDynamica).
3. If your change alters output for existing tests, explain why the old
   numbers were wrong or incomplete.

## Commit style

- Imperative mood: "Add Redfield relaxation superoperator," not
  "Added" / "Adds" / "Adding."
- Reference issues where relevant: `Fixes #42`.
- Keep the subject line under ~70 characters; use the body for "why."

## Filing issues

Two kinds of issues are most useful:

- **Bug reports** with a minimal reproducer (ideally a standalone Rust
  snippet that panics / returns the wrong number).
- **Design discussions** — open these *before* writing large amounts of
  code. The project is small enough that a 10-minute back-and-forth can
  save a weekend.

## Code of conduct

Be decent. If something feels off, contact the maintainer (see `Cargo.toml`).

## License

By submitting a contribution you agree that it will be dual-licensed under
MIT and Apache-2.0 per the terms in the repo root.
