use std::path::{Path, PathBuf};

use crate::{
    backends::universal::UniversalBackend,
    traits::{ActiveTask, CancelOutcome, DownloadBackend},
    util::runtime::TaskJoinHandle,
};

pub struct UniversalActiveTask {
    task_handles: Box<[Box<dyn TaskJoinHandle<()>>]>,
    resume_artifact_path: PathBuf,
}

impl std::fmt::Debug for UniversalActiveTask {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        formatter
            .debug_struct("UniversalActiveTask")
            .field("resume_artifact_path", &self.resume_artifact_path)
            .finish_non_exhaustive()
    }
}

impl UniversalActiveTask {
    pub fn new(
        task_handles: Box<[Box<dyn TaskJoinHandle<()>>]>,
        resume_artifact_path: PathBuf,
    ) -> Self {
        Self {
            task_handles,
            resume_artifact_path,
        }
    }
}

impl Drop for UniversalActiveTask {
    fn drop(&mut self) {
        for handle in self.task_handles.iter() {
            handle.abort();
        }
    }
}

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
impl ActiveTask for UniversalActiveTask {
    type Backend = UniversalBackend;

    async fn pause(
        mut self,
        _destination: &Path,
    ) -> Result<PathBuf, <Self::Backend as DownloadBackend>::Error> {
        for task_handle in std::mem::take(&mut self.task_handles) {
            task_handle.abort();
        }
        Ok(std::mem::take(&mut self.resume_artifact_path))
    }

    async fn cancel(
        mut self,
        _destination: &Path,
    ) -> CancelOutcome {
        for task_handle in std::mem::take(&mut self.task_handles) {
            task_handle.abort();
        }
        CancelOutcome::BestEffort
    }
}
