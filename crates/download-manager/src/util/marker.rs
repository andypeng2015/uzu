//! Thread-safety marker traits that collapse to nothing on wasm.
//!
//! Native targets run on tokio's multithreaded runtime, so futures and shared values must be
//! `Send`/`Sync`. The wasm target runs single-threaded on `spawn_local` over OPFS/`JsFuture`, which
//! are fundamentally `!Send`. These aliases resolve to `Send`/`Sync` on native and to an empty
//! blanket bound on wasm, letting the same trait and spawn signatures compile on both targets.

#[cfg(not(target_family = "wasm"))]
pub trait MaybeSend: Send {}
#[cfg(not(target_family = "wasm"))]
impl<T: Send + ?Sized> MaybeSend for T {}

#[cfg(target_family = "wasm")]
pub trait MaybeSend {}
#[cfg(target_family = "wasm")]
impl<T: ?Sized> MaybeSend for T {}

#[cfg(not(target_family = "wasm"))]
pub trait MaybeSync: Sync {}
#[cfg(not(target_family = "wasm"))]
impl<T: Sync + ?Sized> MaybeSync for T {}

#[cfg(target_family = "wasm")]
pub trait MaybeSync {}
#[cfg(target_family = "wasm")]
impl<T: ?Sized> MaybeSync for T {}
