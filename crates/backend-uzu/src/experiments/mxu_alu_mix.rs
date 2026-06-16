//! Hybrid MXU + SIMD/ALU GEMM experiment (`gemm_mixed.metal`).
//!
//! Runs one BF16 full-precision GEMM as a single dispatch where each 64x64
//! output tile is computed on either the MXU tensor pipeline or the SIMD/ALU
//! pipeline, interleaved across the grid by a tunable split. `MixedGemm` is a
//! public, opaque handle so a dinghy-runnable bench target can drive the kernel
//! without depending on the crate's `#[cfg(test)]` helpers.

use half::bf16;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Allocation, Encoder,
            gpu_types::{
                gemm::{GemmAlignment, GemmDTransform},
                matmul::GemmParams,
            },
        },
        metal::{GemmMixedMetalKernel, Metal, MetalContext},
    },
};

/// Output tile size shared by both pipelines (block_m == block_n).
pub const MIXED_TILE: u32 = 64;
/// K-block of the MXU tiling (`Tile64x64x256`).
pub const MXU_BLOCK_K: u32 = 256;
/// K-block of the SIMD tiling (`Tile64x64x32`).
pub const SIMD_BLOCK_K: u32 = 32;

fn mixed_params(
    m: u32,
    n: u32,
    k: u32,
) -> [GemmParams; 2] {
    let base = GemmParams {
        M: m,
        N: n,
        K: k,
        leading_dimension_a: k,
        leading_dimension_b: k, // B is [N, K], transposed
        leading_dimension_d: n,
        threadgroups_per_column: m.div_ceil(MIXED_TILE),
        threadgroups_per_row: n.div_ceil(MIXED_TILE),
        aligned_inner_iterations: 0,
        use_morton: false,
        ab_scale: 1.0,
    };
    [
        GemmParams {
            aligned_inner_iterations: k / MXU_BLOCK_K,
            ..base
        },
        GemmParams {
            aligned_inner_iterations: k / SIMD_BLOCK_K,
            ..base
        },
    ]
}

/// Opaque handle to the hybrid GEMM kernel specialized for a problem shape.
pub struct MixedGemm {
    kernel: GemmMixedMetalKernel,
    params: [GemmParams; 2],
    m: u32,
    n: u32,
}

impl MixedGemm {
    /// Build the kernel for `[M, K] x [N, K]^T -> [M, N]` in BF16. Alignment is
    /// derived from the shape using the MXU K-block (the conservative choice:
    /// `k % 256 == 0` implies `k % 32 == 0`, so both pipelines stay correct).
    pub fn new(
        context: &MetalContext,
        m: u32,
        n: u32,
        k: u32,
    ) -> Self {
        let alignment = GemmAlignment::new(m % MIXED_TILE == 0, n % MIXED_TILE == 0, k % MXU_BLOCK_K == 0);
        let kernel = GemmMixedMetalKernel::new(
            context,
            bf16::data_type(),
            bf16::data_type(),
            bf16::data_type(),
            GemmDTransform::empty(),
            alignment,
        )
        .expect("GemmMixed kernel");
        Self {
            kernel,
            params: mixed_params(m, n, k),
            m,
            n,
        }
    }

    /// Encode one hybrid GEMM. `a` is `[M, K]`, `b` is `[N, K]` (transposed),
    /// `d` is `[M, N]`. A tile at linear grid index `i` runs on the MXU when
    /// `(i % tile_period) < mxu_tiles_per_period`, else on the SIMD/ALU path.
    /// `tile_period == mxu_tiles_per_period` => all MXU; `mxu_tiles_per_period
    /// == 0` => all SIMD.
    pub fn encode(
        &self,
        a: &Allocation<Metal>,
        b: &Allocation<Metal>,
        d: &mut Allocation<Metal>,
        tile_period: u32,
        mxu_tiles_per_period: u32,
        encoder: &mut Encoder<Metal>,
    ) {
        self.kernel.encode(
            (a, 0usize),
            (b, 0usize),
            &mut *d,
            &self.params[..],
            self.n.div_ceil(MIXED_TILE),
            self.m.div_ceil(MIXED_TILE),
            1,
            tile_period,
            mxu_tiles_per_period,
            encoder,
        );
    }
}
