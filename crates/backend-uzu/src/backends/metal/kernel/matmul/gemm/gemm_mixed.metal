// Experimental hybrid GEMM kernel that mixes MXU (matrix unit) and SIMD/ALU
// computation inside a SINGLE dispatch. Each threadgroup computes one 64x64
// output tile and chooses, at runtime, whether to run it on the MXU tensor
// pipeline (`MxuMmaCore`, Tile64x64x256) or the SIMD/ALU pipeline
// (`SimdgroupMmaCore`, Tile64x64x32) based on a tunable split parameter.
//
// Both tilings share the same output-tile geometry (64x64) and the same 2x2
// simdgroup layout (128 threads/threadgroup), so a single grid + threadgroup
// shape serves both cores. Tiles are interleaved across the grid so that, on a
// given GPU core, MXU-tile and ALU-tile threadgroups co-reside and the two
// execution pipelines run in parallel.
//
// Full-precision BF16/F32 only (no quantization, no output transforms beyond
// what the caller specializes). This is a perf experiment, not a production
// path: see `gemm.metal` for the production single-pipeline kernel.

#include "../../common/dsl.h"
#include "../../common/defines.h"
#include "../../common/thread_context.h"
#include "../generated/gemm.h"

#include "common/gemm_tiling.h"
#include "common/mxu_mma_core.h"
#include "common/simdgroup_mma_core.h"

using namespace metal;
using namespace uzu::gemm;

// Both cores produce a 64x64 tile with 2x2 simdgroups; they differ only in the
// K-block (256 for MXU, 32 for SIMD) and in threadgroup-memory usage. The MXU
// full-precision path does not touch threadgroup memory, so sizing the shared
// arrays to the SIMD path's needs is sufficient for both.
#define GEMM_MIXED_SIMD_TILING GemmTiling::Tile64x64x32_Simdgroups2x2
#define GEMM_MIXED_MXU_TILING GemmTiling::Tile64x64x256_Simdgroups2x2
#define GEMM_MIXED_TGA_ELEMENTS                                                                                        \
  (gemm_tiling_block_m(GEMM_MIXED_SIMD_TILING) * (gemm_tiling_block_k(GEMM_MIXED_SIMD_TILING) + 16 / int(sizeof(AT))))
#define GEMM_MIXED_TGB_ELEMENTS                                                                                        \
  (gemm_tiling_block_n(GEMM_MIXED_SIMD_TILING) * (gemm_tiling_block_k(GEMM_MIXED_SIMD_TILING) + 16 / int(sizeof(BT))))

template <typename AT, typename BT, typename DT>
VARIANTS(AT, bfloat)
VARIANTS(BT, bfloat)
VARIANTS(DT, bfloat)
KERNEL(GemmMixed)(
    const device AT* a,
    const device BT* b,
    device DT* d,
    // params[0] drives the MXU tiles (aligned_inner_iterations = K / 256),
    // params[1] drives the SIMD tiles (aligned_inner_iterations = K / 32).
    const constant uzu::matmul::GemmParams* params,
    const constant uint& group_count_x,
    const constant uint& group_count_y,
    const constant uint& group_count_z,
    // Interleave control: a tile at linear grid index `i` runs on the MXU when
    // `(i % tile_period) < mxu_tiles_per_period`, otherwise on the SIMD/ALU
    // pipeline. tile_period == mxu_tiles_per_period => all MXU;
    // mxu_tiles_per_period == 0 => all SIMD.
    const constant uint& tile_period,
    const constant uint& mxu_tiles_per_period,
    const GemmDTransform output_transform SPECIALIZE,
    const GemmAlignment alignment SPECIALIZE,
    threadgroup AT a_shared[GEMM_MIXED_TGA_ELEMENTS],
    threadgroup BT b_shared[GEMM_MIXED_TGB_ELEMENTS],
    const uint group_x GROUPS(group_count_x),
    const uint group_y GROUPS(group_count_y),
    const uint group_z GROUPS(group_count_z),
    const uint thread_x THREADS(METAL_SIMD_SIZE),
    const uint thread_y THREADS(2),
    const uint thread_z THREADS(2),
    const ThreadContext thread_context
) {
  (void)group_x;
  (void)group_y;
  (void)group_z;
  (void)thread_x;
  (void)thread_y;
  (void)thread_z;

  const uint tile_x = thread_context.threadgroup_position.x;
  const uint tile_y = thread_context.threadgroup_position.y;
  const uint linear_tile = tile_y * group_count_x + tile_x;
  const uint period = max(tile_period, 1u);
  const bool use_mxu = (linear_tile % period) < mxu_tiles_per_period;

  if (use_mxu) {
    MxuMmaCore<AT, BT, DT, GEMM_MIXED_MXU_TILING, true, GemmBPrologueKind::FullPrecision, 0, 0>::run(
        a,
        b,
        d,
        &params[0],
        alignment,
        output_transform,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        b_shared,
        thread_context
    );
  } else {
    SimdgroupMmaCore<AT, BT, DT, GEMM_MIXED_SIMD_TILING, true, GemmBPrologueKind::FullPrecision, 0, 0>::run(
        a,
        b,
        d,
        &params[1],
        alignment,
        output_transform,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        a_shared,
        b_shared,
        thread_context
    );
  }
}
