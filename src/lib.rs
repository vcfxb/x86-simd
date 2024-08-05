#![doc = include_str!("../README.md")]
#![deny(missing_copy_implementations, missing_debug_implementations)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(clippy::cast_possible_truncation)]
#![warn(missing_docs)]

// Allow using Sealed trait bounds to limit how much people can shoot themselves in the foot with this crate.
#![allow(private_bounds)]

// Allow unreachable code as parts of this library (especially fall-back code) will become unreachable when
// certain compiler flags are enabled.
#![allow(unreachable_code)]

#![allow(clippy::missing_transmute_annotations)]

#![cfg_attr(not(feature = "std"), no_std)]

// Compiler directive to get docs.rs (which uses the nightly version of the rust compiler) to show
// info about feature required for various modules and functionality.
//
// See: <https://stackoverflow.com/a/70914430>.
#![cfg_attr(all(doc, CHANNEL_NIGHTLY), feature(doc_auto_cfg))]

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
core::compile_error!("The x86-simd crate only supports x86 and x86_64 architectures");

/// A version of [core::hint::unreachable_unchecked] that's only unchecked on release builds, defaulting to
/// [`core::unreachable`] on debug.
///
/// This is used throughout this crate.
///
/// # Safety
/// This is unsafe for the same reasons as [core::hint::unreachable_unchecked].
pub const unsafe fn unreachable_uncheched_on_release() -> ! {
    #[cfg(debug_assertions)]
    core::unreachable!();

    #[cfg(not(debug_assertions))]
    core::hint::unreachable_unchecked()
}

mod sealed;

pub mod integers;
