#[cfg(not(target_family = "wasm"))]
mod tokio;
#[cfg(target_family = "wasm")]
mod wasm;

#[cfg(not(target_family = "wasm"))]
pub use tokio::TokioRuntimeHandle as RuntimeHandle;
#[cfg(target_family = "wasm")]
pub use wasm::WasmRuntimeHandle as RuntimeHandle;

use crate::util::marker::MaybeSend;

pub trait TaskJoinHandle<T>: Send + Sync {
    fn abort(&self);
}

pub fn spawn<F: Future<Output: MaybeSend + 'static> + MaybeSend + 'static>(
    future: F
) -> Box<dyn TaskJoinHandle<F::Output>> {
    #[cfg(not(target_family = "wasm"))]
    return tokio::spawn(future);
    #[cfg(target_family = "wasm")]
    wasm::spawn(future)
}

#[cfg(not(target_family = "wasm"))]
pub async fn run_blocking<T: Send + 'static>(func: impl FnOnce() -> T + Send + 'static) -> T {
    ::tokio::task::spawn_blocking(func).await.expect("blocking task panicked")
}

#[cfg(target_family = "wasm")]
pub async fn run_blocking<T>(func: impl FnOnce() -> T) -> T {
    func()
}
