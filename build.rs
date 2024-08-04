//! Build script to store a CFG flag when we're on nightly so that we can show feature gates on docs.rs.

use rustc_version::{version_meta, Channel};

fn main() {
    println!("cargo::rustc-check-cfg=cfg(CHANNEL_NIGHTLY)");
    if version_meta().unwrap().channel == Channel::Nightly {
        println!("cargo:rustc-cfg=CHANNEL_NIGHTLY");
    }
}
