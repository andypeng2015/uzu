//! Public harness for one-off kernel experiments.
//!
//! Experiment code that benches/tests need to drive on a physical device must
//! be reachable from a dedicated bench target (which only sees the crate's
//! public API, not the `#[cfg(test)]` tree). These thin wrappers expose exactly
//! that surface while keeping the underlying kernels crate-private.

pub mod mxu_alu_mix;
