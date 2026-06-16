mod backend;
mod buffer;
mod command_buffer;
mod context;
mod dense_buffer;
mod device_tier;
mod error;
mod kernel;
mod metal_extensions;
mod sparse;

pub use backend::Metal;
pub use context::MetalContext;
// Experimental hybrid MXU+ALU GEMM kernel. Wrapped by `crate::experiments`.
pub(crate) use kernel::GemmMixedMetalKernel;
pub use kernel::matmul::GemmDispatchPath;
