use std::io;

use opfs::{
    CreateWritableOptions, DirectoryHandle, FileHandle, GetFileHandleOptions, WritableFileStream, WriteCommandType,
    WriteParams,
};

use super::PartFile;
use crate::util::opfilesystem::{get_root_dir, js_value_to_error, resolve_parent};

pub struct OpfsPartFile {
    writer: opfs::persistent::WritableFileStream,
    position: usize,
}

impl OpfsPartFile {
    pub async fn new(
        path: &str,
        resume_from_bytes: u64,
    ) -> io::Result<Self> {
        let root = get_root_dir().await?;
        let (dir, name) = resolve_parent(&root, path, true).await?;
        let mut file = dir
            .get_file_handle_with_options(
                &name,
                &GetFileHandleOptions {
                    create: true,
                },
            )
            .await
            .map_err(js_value_to_error)?;
        let writer = file
            .create_writable_with_options(&CreateWritableOptions {
                keep_existing_data: resume_from_bytes > 0,
            })
            .await
            .map_err(js_value_to_error)?;
        Ok(Self {
            writer,
            position: resume_from_bytes as usize,
        })
    }
}

#[async_trait::async_trait(?Send)]
impl PartFile for OpfsPartFile {
    async fn write_all(
        &mut self,
        buf: &[u8],
    ) -> io::Result<()> {
        self.writer
            .write_with_params(&WriteParams {
                command_type: WriteCommandType::Write,
                data: Some(buf.to_vec()),
                position: Some(self.position),
                size: None,
            })
            .await
            .map_err(js_value_to_error)?;
        self.position += buf.len();
        Ok(())
    }

    async fn flush(&mut self) -> io::Result<()> {
        self.writer.close().await.map_err(js_value_to_error)
    }
}
