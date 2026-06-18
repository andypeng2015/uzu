use tokio::{runtime::Handle, task::JoinHandle};

use crate::util::runtime::TaskJoinHandle;

#[derive(Clone, Debug)]
pub struct TokioRuntimeHandle {
    handle: Handle,
}

impl TokioRuntimeHandle {
    pub fn current() -> Self {
        TokioRuntimeHandle {
            handle: Handle::current(),
        }
    }

    pub fn spawn<F: Future<Output: Send + 'static> + Send + 'static>(
        &self,
        future: F,
    ) -> Box<dyn TaskJoinHandle<F::Output>> {
        Box::new(self.handle.spawn(future))
    }
}

impl<T: Send> TaskJoinHandle<T> for JoinHandle<T> {
    fn abort(&self) {
        self.abort();
    }
}

pub fn spawn<F: Future<Output: Send + 'static> + Send + 'static>(future: F) -> Box<dyn TaskJoinHandle<F::Output>> {
    Box::new(tokio::spawn(future))
}
