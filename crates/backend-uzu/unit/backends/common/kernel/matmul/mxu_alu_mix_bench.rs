#![cfg(metal_backend)]

//! Experiment: mix MXU and SIMD/ALU computation inside ONE GEMM dispatch and
//! measure throughput as a function of how the output tiles are split between
//! the two pipelines. Endpoints are the honest baselines computed by the *same*
//! kernel: `mxu_16_of_16` is all-MXU, `mxu_0_of_16` is all-SIMD/ALU; a win is
//! any interior split that beats all-MXU.
//!
//! MXU-only hardware (A19 / iPhone 17 Pro Max); self-skips elsewhere. The
//! device-runnable copy of this bench lives in `benches/kernel.rs`; both drive
//! `crate::experiments::mxu_alu_mix::MixedGemm`.

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use proc_macros::uzu_bench;

use crate::{
    array::{ArrayContextExt, ArrayElement},
    backends::{
        common::{AllocationType, Context},
        metal::Metal,
    },
    experiments::mxu_alu_mix::MixedGemm,
    tests::{
        matmul::{bench_fp_gemm_shapes, iter_encode_loop},
        util::type_short_name,
    },
};

const TILE_PERIOD: u32 = 16;
const MXU_PER_PERIOD: &[u32] = &[16, 14, 12, 8, 4, 2, 0];

#[uzu_bench]
fn bench_mxu_alu_mix(c: &mut Criterion) {
    let context = crate::tests::util::shared_metal_context();
    if !context.supports_mxu() {
        return;
    }

    let mut group = c.benchmark_group(format!("{}/Kernel/Matmul/MxuAluMix", type_short_name::<Metal>()));

    for shape in bench_fp_gemm_shapes() {
        let (m, k, n) = (shape.m as u32, shape.k as u32, shape.n as u32);
        let mixed = MixedGemm::new(&context, m, n, k);

        let a = context
            .create_allocation(shape.m * shape.k * std::mem::size_of::<bf16>(), AllocationType::Global)
            .expect("a allocation");
        let b = context.create_array_uninitialized(&[shape.n, shape.k], bf16::data_type());
        let mut d = context
            .create_allocation(shape.m * shape.n * std::mem::size_of::<bf16>(), AllocationType::Global)
            .expect("d allocation");

        group.throughput(Throughput::Elements(2 * (m as u64) * (k as u64) * (n as u64)));

        for &mxu_per in MXU_PER_PERIOD {
            let label = format!("{shape}/mxu_{mxu_per}_of_{TILE_PERIOD}");
            group.bench_function(BenchmarkId::new("BF16", label), |bencher| {
                iter_encode_loop::<Metal, _>(&context, bencher, |encoder| {
                    mixed.encode(&a, b.allocation(), &mut d, TILE_PERIOD, mxu_per, encoder);
                });
            });
        }
    }
}
