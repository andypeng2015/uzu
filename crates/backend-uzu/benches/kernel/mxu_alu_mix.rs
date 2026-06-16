//! Device-runnable MXU + SIMD/ALU mixed-GEMM benchmark (companion to the
//! `#[cfg(test)]` lib bench `unit/.../mxu_alu_mix_bench.rs`; both drive
//! `backend_uzu::experiments::mxu_alu_mix::MixedGemm`).
//!
//! Sweeps `mxu_tiles_per_period` over a fixed `TILE_PERIOD`: the endpoints are
//! the honest baselines from the *same* kernel — `mxu_16_of_16` is all-MXU,
//! `mxu_0_of_16` is all-SIMD/ALU — and a win is any interior split that beats
//! all-MXU. MXU-only hardware; self-skips elsewhere.

use std::mem::size_of;

use backend_uzu::{
    backends::{
        common::{AllocationType, Context, Encoder},
        metal::{Metal, MetalContext},
    },
    experiments::mxu_alu_mix::MixedGemm,
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use proc_macros::uzu_bench;

const TILE_PERIOD: u32 = 16;
const MXU_PER_PERIOD: &[u32] = &[16, 14, 12, 8, 4, 2, 0];
// (m, k, n) — compute-bound prefill shapes where the MXU dominates and there is
// ALU headroom to fill. Mirrors `bench_fp_gemm_shapes()`.
const SHAPES: &[(usize, usize, usize)] =
    &[(128, 2048, 8192), (128, 4096, 14336), (256, 4096, 4096), (512, 8192, 2048)];

#[uzu_bench]
fn bench_mxu_alu_mix(c: &mut Criterion) {
    let context = MetalContext::new().expect("Metal context");
    if !context.supports_mxu() {
        return;
    }

    let mut group = c.benchmark_group("Metal/Kernel/Matmul/MxuAluMix");

    for &(m, k, n) in SHAPES {
        let (mu, ku, nu) = (m as u32, k as u32, n as u32);
        let mixed = MixedGemm::new(&context, mu, nu, ku);

        let a = context.create_allocation(m * k * size_of::<bf16>(), AllocationType::Global).expect("a allocation");
        let b = context.create_allocation(n * k * size_of::<bf16>(), AllocationType::Global).expect("b allocation");
        let mut d = context.create_allocation(m * n * size_of::<bf16>(), AllocationType::Global).expect("d allocation");

        group.throughput(Throughput::Elements(2 * (m as u64) * (k as u64) * (n as u64)));

        for &mxu_per in MXU_PER_PERIOD {
            let label = format!("M[{m}]K[{k}]N[{n}]/mxu_{mxu_per}_of_{TILE_PERIOD}");
            group.bench_function(BenchmarkId::new("BF16", label), |bencher| {
                bencher.iter_custom(|iters| {
                    let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
                    for _ in 0..iters {
                        mixed.encode(&a, &b, &mut d, TILE_PERIOD, mxu_per, &mut encoder);
                    }
                    encoder.end_encoding().submit().wait_until_completed().expect("submit").gpu_execution_time()
                });
            });
        }
    }
}
