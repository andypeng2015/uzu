use std::{io, path::Path};

pub async fn create_dir(path: impl AsRef<Path>) -> io::Result<()> {
    create_dir_all(path).await
}

pub async fn create_dir_all(path: impl AsRef<Path>) -> io::Result<()> {
    #[cfg(target_family = "wasm")]
    return super::opfilesystem::create_dir_all(path.as_ref().to_str().unwrap()).await;

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::create_dir_all(path).await
}

pub async fn file_exists(path: impl AsRef<Path>) -> io::Result<bool> {
    #[cfg(target_family = "wasm")]
    return super::opfilesystem::file_exists(path.as_ref().to_str().unwrap()).await;

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::try_exists(path.as_ref()).await
}

pub async fn file_length(path: impl AsRef<Path>) -> io::Result<u64> {
    #[cfg(target_family = "wasm")]
    return super::opfilesystem::file_length(path.as_ref().to_str().unwrap()).await;

    #[cfg(not(target_family = "wasm"))]
    {
        let metadata = tokio::fs::metadata(path.as_ref()).await?;
        Ok(metadata.len())
    }
}

pub async fn hard_link(
    original: impl AsRef<Path>,
    link: impl AsRef<Path>,
) -> io::Result<()> {
    // OPFS doesn't support linking, that's why just copy
    #[cfg(target_family = "wasm")]
    return super::opfilesystem::copy(original.as_ref().to_str().unwrap(), link.as_ref().to_str().unwrap()).await;

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::hard_link(original.as_ref(), link.as_ref()).await
}

pub async fn read(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
    #[cfg(target_family = "wasm")]
    return super::opfilesystem::read(path.as_ref().to_str().unwrap()).await;

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::read(path.as_ref()).await
}

pub async fn read_to_string(path: impl AsRef<Path>) -> io::Result<String> {
    #[cfg(target_family = "wasm")]
    {
        let content = read(path).await?;
        let string = String::from_utf8(content).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
        Ok(string)
    }

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::read_to_string(path.as_ref()).await
}

pub async fn remove_file(path: impl AsRef<Path>) -> io::Result<()> {
    #[cfg(target_family = "wasm")]
    return super::opfilesystem::remove_file(path.as_ref().to_str().unwrap()).await;

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::remove_file(path.as_ref()).await
}

pub async fn rename(
    from: impl AsRef<Path>,
    to: impl AsRef<Path>,
) -> io::Result<()> {
    #[cfg(target_family = "wasm")]
    {
        super::opfilesystem::copy(from.as_ref().to_str().unwrap(), to.as_ref().to_str().unwrap()).await?;
        super::opfilesystem::remove_file(from.as_ref().to_str().unwrap()).await?;
        return Ok(());
    }

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::rename(from, to.as_ref()).await
}

pub async fn write(
    path: impl AsRef<Path>,
    contents: impl AsRef<[u8]>,
) -> io::Result<()> {
    #[cfg(target_family = "wasm")]
    return super::opfilesystem::write(path.as_ref().to_str().unwrap(), contents.as_ref()).await;

    #[cfg(not(target_family = "wasm"))]
    tokio::fs::write(path, contents).await
}

pub async fn write_with_sync_all(
    path: impl AsRef<Path>,
    contents: impl AsRef<[u8]>,
) -> io::Result<()> {
    #[cfg(target_family = "wasm")]
    return write(path, contents).await;

    #[cfg(not(target_family = "wasm"))]
    {
        use tokio::io::AsyncWriteExt;

        let mut file = tokio::fs::OpenOptions::new().write(true).create_new(true).open(&path).await?;
        file.write_all(contents.as_ref()).await?;
        file.sync_all().await
    }
}
