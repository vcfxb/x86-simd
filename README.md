# x86-simd
This crate provides safe (and some unsafe) interfaces to (some) x86 (and x86_64) SIMD intrinsics.
This crate is not exhaustive about what it does but is rather a "best-effort" to support/wrap many commonly used 
SSE2+ and AVX types and functions.

This crate is built on the stable version of the rust compiler and standard library and does not required nightly 
portable SIMD features.

This crate has zero dependencies and it can be built with or without the standard library. This crate does not use 
the `alloc` crate. Building with the standard library (enabling the `std` feature, which is on by default) allows 
this crate to detect CPU features at run-time, which can be incredibly useful when compiling without knowing exactly
which SIMD features will be available on a user's machine. Without the `std` feature, this crate relies entirely on the 
features that the compiler lists as available. This can be configured using flags to `rustc` -- see 
<https://rust-lang.github.io/packed_simd/perf-guide/target-feature/rustflags.html>.

If you attempt to compile or depend on this crate on an architecture that is not `x86` or `x86_64`, this crate will 
produce a compiler error. 

This crate may be depended on conditionally by moving it to the
`[target.'cfg(any(target_arch = "x86", target_arch = "x86_64"))'.dependencies]` section in your `Cargo.toml`
and making any reference to it in your code `#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]`.

### Badges
| Service | Badge |
|:---:|:---:|
| Cargo Check Status | ![Cargo Check status](https://github.com/vcfxb/x86-simd/actions/workflows/cargo-check.yml/badge.svg?branch=main) |
| Cargo Test Status | ![Cargo Test status](https://github.com/vcfxb/x86-simd/actions/workflows/cargo-test.yml/badge.svg?branch=main) |
| Cargo Clippy Status | ![Cargo Clippy status](https://github.com/vcfxb/x86-simd/actions/workflows/cargo-clippy.yml/badge.svg?branch=main) |
| Code Coverage (Coveralls) | [![Coverage Status](https://coveralls.io/repos/github/vcfxb/x86-simd/badge.svg?branch=main)](https://coveralls.io/github/vcfxb/x86-simd?branch=main) |
| Code Coverage (Codecov.io) | [![codecov](https://codecov.io/github/vcfxb/x86-simd/branch/main/graph/badge.svg?token=HO07JEYMIH)](https://codecov.io/github/vcfxb/x86-simd/commits?branch=main) |
| Docs.rs | [![Documentation](https://docs.rs/x86-simd/badge.svg)](https://docs.rs/x86-simd) |
| Crates.io | [![Crates.io](https://img.shields.io/crates/v/x86-simd.svg)](https://crates.io/crates/x86-simd) |

|  | Downloads|
|:---:|:---:|
| Crates.io | [![Crates.io](https://img.shields.io/crates/d/x86-simd.svg)](https://crates.io/crates/x86-simd) |
<!-- | Crates.io (Latest) | [![Crates.io](https://img.shields.io/crates/dv/x86-simd.svg)](https://crates.io/crates/x86-simd/0.1.0) | -->

