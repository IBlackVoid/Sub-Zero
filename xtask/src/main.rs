//! `cargo xtask` — workspace task runner.
//!
//! The point: every checked-in verification step (smoke tests, dep
//! audit, doc-warnings build) runs through one cross-platform Rust
//! binary. No more "this works on the maintainer's box but not on
//! Windows CI" because `bash` isn't available.
//!
//! Subcommands:
//!
//! - `smoke` — fast cross-platform smoke: format, clippy, doc-warnings,
//!   workspace tests, the CLI integration sub-set that doesn't need the
//!   CC0 fixture audio. Mirrors `scripts/ci/run_integration_smoke.sh`
//!   for the parts that can run on a vanilla developer machine.
//!
//! - `verify` — the full gauntlet: smoke + `cargo deny check` +
//!   `cargo audit`. Slower; used by CI's release gate.
//!
//! Invocation: `cargo xtask <subcommand>`. The `cargo` alias in
//! `.cargo/config.toml` rewrites that to
//! `cargo run --package xtask --quiet -- <subcommand>` so the shim is
//! invisible.

use std::env;
use std::path::Path;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    match cmd {
        "smoke" => run_smoke(),
        "verify" => run_verify(),
        "" | "-h" | "--help" => {
            print_usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("xtask: unknown subcommand '{other}'\n");
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    let bin = env::args().next().unwrap_or_else(|| "xtask".to_string());
    println!("usage: {bin} <smoke|verify>\n");
    println!("  smoke   fmt + clippy + doc-warnings + workspace tests");
    println!("  verify  smoke + cargo deny check + cargo audit");
}

// `cargo xtask smoke`: the cross-platform replacement for
// `scripts/ci/run_integration_smoke.sh`. The bash script's
// ffmpeg/neural/quality-eval stages are intentionally not ported —
// those require the CC0 fixture (`fixtures/clip_10s.wav`) which still
// isn't in the tree (see ROADMAP item 6). When the fixture lands, this
// runner gains the same steps gated on `fixtures/clip_10s.wav` existing.
fn run_smoke() -> ExitCode {
    println!("[xtask] cargo fmt --all --check");
    if !cargo(&["fmt", "--all", "--", "--check"]) {
        return ExitCode::FAILURE;
    }
    println!("[xtask] cargo clippy --workspace --all-targets -D warnings");
    if !cargo(&[
        "clippy",
        "--workspace",
        "--all-targets",
        "--",
        "-D",
        "warnings",
    ]) {
        return ExitCode::FAILURE;
    }
    println!("[xtask] cargo doc --workspace --no-deps (rustdoc warnings denied)");
    let doc_env: &[(&str, &str)] = &[("RUSTDOCFLAGS", "-D warnings")];
    if !cargo_with_env(&["doc", "--workspace", "--no-deps"], doc_env) {
        return ExitCode::FAILURE;
    }
    println!("[xtask] cargo test --workspace --locked");
    if !cargo(&["test", "--workspace", "--locked"]) {
        return ExitCode::FAILURE;
    }
    // Selectively run the CLI integration sub-set that doesn't need
    // ffmpeg or a Python MT daemon — those are gated by env vars in
    // tests/cli_integration.rs and stay off here.
    println!("[xtask] cargo test --test cli_integration");
    if !cargo(&["test", "--test", "cli_integration"]) {
        return ExitCode::FAILURE;
    }

    // Stretch: if the CC0 fixture has landed, run the deeper smoke.
    let fixture = workspace_root().join("fixtures").join("clip_10s.wav");
    if fixture.is_file() {
        println!("[xtask] fixtures/clip_10s.wav present — running ffmpeg+neural smoke");
        if !cargo_with_env(
            &["test", "--test", "cli_integration", "ffmpeg_ffprobe_smoke"],
            &[("VOIDEX_RUN_FFMPEG_SMOKE", "1")],
        ) {
            return ExitCode::FAILURE;
        }
    } else {
        println!("[xtask] fixtures/clip_10s.wav absent — skipping ffmpeg/neural smoke");
    }

    println!("[xtask] smoke ok");
    ExitCode::SUCCESS
}

fn run_verify() -> ExitCode {
    if !matches!(run_smoke(), ExitCode::SUCCESS) {
        return ExitCode::FAILURE;
    }
    println!("[xtask] cargo deny check");
    if !cargo(&["deny", "check"]) {
        eprintln!(
            "[xtask] cargo-deny is required for `verify`. install with: cargo install cargo-deny"
        );
        return ExitCode::FAILURE;
    }
    println!("[xtask] cargo audit");
    if !cargo(&["audit"]) {
        eprintln!(
            "[xtask] cargo-audit is required for `verify`. install with: cargo install cargo-audit"
        );
        return ExitCode::FAILURE;
    }
    println!("[xtask] verify ok");
    ExitCode::SUCCESS
}

fn cargo(args: &[&str]) -> bool {
    cargo_with_env(args, &[])
}

fn cargo_with_env(args: &[&str], env_pairs: &[(&str, &str)]) -> bool {
    let cargo_bin = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut cmd = Command::new(cargo_bin);
    cmd.args(args).current_dir(workspace_root());
    for (k, v) in env_pairs {
        cmd.env(k, v);
    }
    match cmd.status() {
        Ok(status) => status.success(),
        Err(e) => {
            eprintln!("[xtask] failed to spawn cargo: {e}");
            false
        }
    }
}

/// Locate the workspace root by walking up from this binary's manifest
/// dir. `CARGO_MANIFEST_DIR` points at `xtask/`; the parent is the
/// workspace root in this layout.
fn workspace_root() -> std::path::PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| std::env::current_dir().expect("cwd"))
}
