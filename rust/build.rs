//! Embed a digest of the package sources, used in resumable batch identities.

use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

fn sources(directory: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = fs::read_dir(directory)
        .expect("source directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().map(|e| e == "rs").unwrap_or(false) {
            out.push(path);
        }
    }
}

fn main() {
    let root = Path::new("src");
    let mut files = Vec::new();
    sources(root, &mut files);
    let mut digest = Sha256::new();
    for file in &files {
        let name = file
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        let content = fs::read(file).unwrap();
        for part in [name.as_bytes(), content.as_slice()] {
            digest.update((part.len() as u64).to_be_bytes());
            digest.update(part);
        }
        println!("cargo:rerun-if-changed={}", file.display());
    }
    println!("cargo:rerun-if-changed=src");
    println!(
        "cargo:rustc-env=HKFACTOR_SOURCE_SHA256={}",
        hex::encode(digest.finalize())
    );
}
