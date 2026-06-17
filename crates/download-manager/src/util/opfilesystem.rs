use std::{io, io::ErrorKind};

use opfs::{
    CreateWritableOptions, DirectoryHandle, FileHandle, GetDirectoryHandleOptions, GetFileHandleOptions,
    WritableFileStream, WriteCommandType, WriteParams, persistent::app_specific_dir,
};
use wasm_bindgen::JsValue;

pub async fn copy(
    from_path: &str,
    to_path: &str,
) -> io::Result<()> {
    if from_path == to_path {
        return Err(io::Error::new(ErrorKind::InvalidFilename, "Paths are equal"));
    }

    let chunk_size = 16 * 1024;
    let root = get_root_dir().await?;

    let (src_dir, src_name) = resolve_parent(&root, from_path, false).await?;
    let src_file = src_dir
        .get_file_handle_with_options(
            &src_name,
            &GetFileHandleOptions {
                create: false,
            },
        )
        .await
        .map_err(js_value_to_error)?;
    let total = src_file.size().await.map_err(js_value_to_error)?;

    let (dst_dir, dst_name) = resolve_parent(&root, to_path, true).await?;
    let mut dst_file = dst_dir
        .get_file_handle_with_options(
            &dst_name,
            &GetFileHandleOptions {
                create: true,
            },
        )
        .await
        .map_err(js_value_to_error)?;
    let mut writer = dst_file
        .create_writable_with_options(&CreateWritableOptions {
            keep_existing_data: false,
        })
        .await
        .map_err(js_value_to_error)?;

    let mut offset = 0usize;
    while offset < total {
        let end = (offset + chunk_size).min(total);
        let chunk = src_file.read_range(offset..end).await.map_err(js_value_to_error)?;
        if chunk.is_empty() {
            break;
        }

        let chunk_length = chunk.len();
        match writer
            .write_with_params(&WriteParams {
                command_type: WriteCommandType::Write,
                data: Some(chunk),
                position: Some(offset),
                size: None,
            })
            .await
        {
            Ok(_) => {},
            Err(err) => {
                writer.close().await.map_err(js_value_to_error)?;
                return Err(js_value_to_error(err));
            },
        }
        offset += chunk_length;
    }

    writer.close().await.map_err(js_value_to_error)?;

    Ok(())
}

pub async fn create_dir_all(path: &str) -> io::Result<()> {
    let mut root = get_root_dir().await?;
    let segments = path.split('/').filter(|s| !s.is_empty());
    for segment in segments {
        root = root
            .get_directory_handle_with_options(
                segment,
                &GetDirectoryHandleOptions {
                    create: true,
                },
            )
            .await
            .map_err(js_value_to_error)?;
    }
    Ok(())
}

pub async fn file_exists(path: &str) -> io::Result<bool> {
    let root = get_root_dir().await?;
    let exists = root
        .get_file_handle_with_options(
            path,
            &GetFileHandleOptions {
                create: false,
            },
        )
        .await
        .is_ok();
    Ok(exists)
}

pub async fn file_length(path: &str) -> io::Result<u64> {
    let root = get_root_dir().await?;
    let file = root
        .get_file_handle_with_options(
            path,
            &GetFileHandleOptions {
                create: false,
            },
        )
        .await
        .map_err(js_value_to_error)?;
    let size = file.size().await.map_err(js_value_to_error)?;
    Ok(size as u64)
}

pub async fn read(path: &str) -> io::Result<Vec<u8>> {
    let root = get_root_dir().await?;
    let file = root
        .get_file_handle_with_options(
            path,
            &GetFileHandleOptions {
                create: false,
            },
        )
        .await
        .map_err(js_value_to_error)?;
    let result = file.read().await.map_err(js_value_to_error)?;
    Ok(result)
}

pub async fn remove_file(path: &str) -> io::Result<()> {
    let root = get_root_dir().await?;
    let (mut parent, name) = resolve_parent(&root, path, false).await?;
    parent.remove_entry(&name).await.map_err(js_value_to_error)?;
    Ok(())
}

pub async fn write(
    path: &str,
    contents: &[u8],
) -> io::Result<()> {
    let root = get_root_dir().await?;
    let mut file = root
        .get_file_handle_with_options(
            path,
            &GetFileHandleOptions {
                create: true,
            },
        )
        .await
        .map_err(js_value_to_error)?;

    let mut writer = file
        .create_writable_with_options(&CreateWritableOptions {
            keep_existing_data: false,
        })
        .await
        .map_err(js_value_to_error)?;
    writer.write_at_cursor_pos(contents).await.map_err(js_value_to_error)?;
    writer.close().await.map_err(js_value_to_error)?;

    Ok(())
}

pub(crate) async fn resolve_parent(
    root: &opfs::persistent::DirectoryHandle,
    path: &str,
    create: bool,
) -> io::Result<(opfs::persistent::DirectoryHandle, String)> {
    let mut parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    let name = parts.pop().ok_or(io::Error::new(ErrorKind::InvalidFilename, "empty path"))?.to_string();

    let mut dir = root.clone();
    for segment in parts {
        dir = dir
            .get_directory_handle_with_options(
                segment,
                &GetDirectoryHandleOptions {
                    create,
                },
            )
            .await
            .map_err(js_value_to_error)?;
    }
    Ok((dir, name))
}

pub(crate) async fn get_root_dir() -> io::Result<opfs::persistent::DirectoryHandle> {
    app_specific_dir().await.map_err(js_value_to_error)
}

pub(crate) fn js_value_to_error(value: JsValue) -> io::Error {
    io::Error::new(ErrorKind::Other, value.as_string().unwrap())
}
