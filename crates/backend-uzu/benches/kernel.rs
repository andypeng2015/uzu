#![feature(custom_test_frameworks)]
#![test_runner(test_runner::uzu_harness)]

// Device-runnable kernel microbenchmarks.
//
// Unlike the `Metal/Kernel/*` benches in the library `#[cfg(test)]` tree (run
// natively via `cargo bench -p backend-uzu --lib`), this is a standalone bench
// target that uses only backend-uzu's public API, so `cargo-dinghy` can build
// its on-device runner from it. See `benches/BENCHMARKS.md`.
//
// NOTE: this file is `include!`d verbatim into dinghy's generated runner, so it
// must contain no inner doc comments (`//!`) or inner attributes beyond the two
// harness attributes above (which dinghy strips).
//
// Run on the iPhone 17 Pro Max (A19) — the MXU-capable device:
//   cargo dinghy -d "$DEVICE" \
//     -e CRITERION_HOME=criterion/a19 \
//     --sync-dirs "$(pwd)/target/criterion=Documents/criterion" \
//     bench -p backend-uzu --bench kernel -- "Metal/Kernel/Matmul/MxuAluMix" \
//     --save-baseline mxu_alu_mix_a19

#[cfg(metal_backend)]
#[path = "kernel/mxu_alu_mix.rs"]
mod mxu_alu_mix;
