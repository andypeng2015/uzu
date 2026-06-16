#![cfg(metal_backend)]

//! Correctness guard for the hybrid MXU+ALU GEMM kernel: a single split GEMM
//! must equal the full CPU reference for every split ratio, including all-MXU,
//! all-SIMD, and interleaved mixes. This proves that splitting output tiles
//! across the two pipelines (disjoint tiles, shared read-only inputs) does not
//! corrupt results before any timing numbers are trusted. MXU hardware only.

use half::bf16;
use proc_macros::uzu_test;

use crate::{
    array::ArrayContextExt,
    backends::{
        common::{AllocationType, Context, Encoder},
        metal::Metal,
    },
    experiments::mxu_alu_mix::MixedGemm,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation_with_data, allocation_to_vec},
        matmul::{Case, Shape, cpu_reference, deterministic_input},
    },
};

#[uzu_test]
fn mxu_alu_mix_matches_reference() {
    let context = crate::tests::util::shared_metal_context();
    if !context.supports_mxu() {
        return;
    }

    // Aligned shape: m, n multiples of 64; k a multiple of 256.
    let shape = Shape::new(128, 512, 256);
    let (m, k, n) = (shape.m as u32, shape.k as u32, shape.n as u32);
    let input = deterministic_input::<bf16>(Case::new(shape));
    let expected = cpu_reference::<bf16>(&input);

    let mixed = MixedGemm::new(&context, m, n, k);

    // (tile_period, mxu_tiles_per_period): all-MXU, all-SIMD, and several mixes.
    let splits = [(1u32, 1u32), (1, 0), (2, 1), (4, 1), (4, 3), (16, 8)];
    for (period, mxu_per) in splits {
        let a = alloc_allocation_with_data::<Metal, bf16>(&context, &input.a);
        let b = context.create_array_from(&[shape.n, shape.k], &input.b);
        let mut d = context
            .create_allocation(shape.m * shape.n * std::mem::size_of::<bf16>(), AllocationType::Global)
            .expect("d allocation");

        let mut encoder = Encoder::<Metal>::new(&context).expect("encoder");
        mixed.encode(&a, b.allocation(), &mut d, period, mxu_per, &mut encoder);
        encoder.end_encoding().submit().wait_until_completed().expect("submit");

        let actual = allocation_to_vec::<Metal, bf16>(&d);
        assert_eq_float(&expected, &actual, 1.0, &format!("MxuAluMix period={period} mxu_per={mxu_per}"));
    }
}
