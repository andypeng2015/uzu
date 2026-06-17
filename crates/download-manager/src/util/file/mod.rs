use std::{io, path::Path};

#[cfg(target_family = "wasm")]
mod opfs_part_file;
#[cfg(not(target_family = "wasm"))]
mod tokio_part_file;

#[cfg(target_family = "wasm")]
use opfs_part_file::OpfsPartFile;
#[cfg(not(target_family = "wasm"))]
use tokio_part_file::TokioPartFile;

#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
pub trait PartFile {
    async fn write_all(
        &mut self,
        buf: &[u8],
    ) -> io::Result<()>;

    async fn flush(&mut self) -> io::Result<()>;
}

impl dyn PartFile {
    pub async fn new(
        path: impl AsRef<Path>,
        resume_from_bytes: u64,
    ) -> io::Result<impl PartFile> {
        #[cfg(target_family = "wasm")]
        return OpfsPartFile::new(path.as_ref().to_str().unwrap(), resume_from_bytes).await;

        #[cfg(not(target_family = "wasm"))]
        TokioPartFile::new(path, resume_from_bytes).await
    }
}
