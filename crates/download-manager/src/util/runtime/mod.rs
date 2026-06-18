#[cfg(not(target_family = "wasm"))]
mod tokio;
#[cfg(target_family = "wasm")]
mod wasm;

#[cfg(not(target_family = "wasm"))]
pub use tokio::TokioRuntimeHandle as RuntimeHandle;
#[cfg(target_family = "wasm")]
pub use wasm::WasmRuntimeHandle as RuntimeHandle;

pub trait TaskJoinHandle<T>: Send + Sync {
    fn abort(&self);
}

pub fn spawn<F: Future<Output: Send + 'static> + Send + 'static>(future: F) -> Box<dyn TaskJoinHandle<F::Output>> {
    #[cfg(not(target_family = "wasm"))]
    return tokio::spawn(future);
    #[cfg(target_family = "wasm")]
    wasm::spawn(future)
}
