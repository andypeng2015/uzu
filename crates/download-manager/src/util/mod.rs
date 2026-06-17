#![allow(dead_code)]

#[cfg(target_family = "wasm")]
mod opfilesystem;

pub mod file;
pub mod filesystem;
