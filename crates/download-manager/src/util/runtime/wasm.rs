use futures_util::future::{AbortHandle, Abortable};
use wasm_bindgen_futures::spawn_local;

use crate::util::runtime::TaskJoinHandle;

#[derive(Clone, Debug)]
pub struct WasmRuntimeHandle;

impl WasmRuntimeHandle {
    pub fn current() -> Self {
        WasmRuntimeHandle
    }

    pub fn spawn<F: Future<Output: 'static> + 'static>(
        &self,
        future: F,
    ) -> Box<dyn TaskJoinHandle<F::Output>> {
        spawn(future)
    }
}

pub struct WasmTaskJoinHandle {
    abort_handle: AbortHandle,
}

impl<T> TaskJoinHandle<T> for WasmTaskJoinHandle {
    fn abort(&self) {
        self.abort_handle.abort();
    }
}

pub fn spawn<F: Future<Output: 'static> + 'static>(future: F) -> Box<dyn TaskJoinHandle<F::Output>> {
    let (abort_handle, abort_registration) = AbortHandle::new_pair();
    let future = Abortable::new(future, abort_registration);
    spawn_local(async move {
        let _ = future.await;
    });
    Box::new(WasmTaskJoinHandle {
        abort_handle,
    })
}
