//! Minimal NumPy ``.npy`` (format 1.0) reader and writer.
//!
//! Panels are persisted in the same layout the Python implementation uses,
//! so caches prepared by either implementation can be read by the other.

use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};

use crate::matrix::Mat;

const MAGIC: &[u8] = b"\x93NUMPY";

/// Element types supported by panel arrays.
pub trait NpyElement: Copy + Default {
    const DESCR: &'static str;
    const SIZE: usize;
    fn read(bytes: &[u8]) -> Self;
    fn write(self, out: &mut Vec<u8>);
}

impl NpyElement for f64 {
    const DESCR: &'static str = "<f8";
    const SIZE: usize = 8;
    fn read(bytes: &[u8]) -> Self {
        f64::from_le_bytes(bytes.try_into().unwrap())
    }
    fn write(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl NpyElement for i64 {
    const DESCR: &'static str = "<i8";
    const SIZE: usize = 8;
    fn read(bytes: &[u8]) -> Self {
        i64::from_le_bytes(bytes.try_into().unwrap())
    }
    fn write(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl NpyElement for i8 {
    const DESCR: &'static str = "|i1";
    const SIZE: usize = 1;
    fn read(bytes: &[u8]) -> Self {
        bytes[0] as i8
    }
    fn write(self, out: &mut Vec<u8>) {
        out.push(self as u8);
    }
}

impl NpyElement for bool {
    const DESCR: &'static str = "|b1";
    const SIZE: usize = 1;
    fn read(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
    fn write(self, out: &mut Vec<u8>) {
        out.push(self as u8);
    }
}

fn header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let shape_text = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut dict =
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_text}, }}");
    // Total header (magic + version + length + dict + newline) is 64-aligned.
    let unpadded = MAGIC.len() + 2 + 2 + dict.len() + 1;
    let padding = (64 - unpadded % 64) % 64;
    dict.push_str(&" ".repeat(padding));
    dict.push('\n');
    let mut out = Vec::with_capacity(10 + dict.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(dict.len() as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out
}

fn save_raw<T: NpyElement>(path: &Path, shape: &[usize], values: &[T]) -> Result<()> {
    let mut out = header(T::DESCR, shape);
    out.reserve(values.len() * T::SIZE);
    for value in values {
        value.write(&mut out);
    }
    let mut file =
        fs::File::create(path).with_context(|| format!("Cannot write {}", path.display()))?;
    file.write_all(&out)?;
    Ok(())
}

pub fn save_vec<T: NpyElement>(path: &Path, values: &[T]) -> Result<()> {
    save_raw(path, &[values.len()], values)
}

pub fn save_mat<T: NpyElement>(path: &Path, matrix: &Mat<T>) -> Result<()> {
    save_raw(path, &[matrix.rows, matrix.cols], &matrix.data)
}

struct Parsed {
    descr: String,
    fortran: bool,
    shape: Vec<usize>,
    offset: usize,
}

fn parse_header(bytes: &[u8]) -> Result<Parsed> {
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        bail!("Not a NumPy array file");
    }
    let major = bytes[6];
    let (length, start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        ),
        _ => bail!("Unsupported NumPy format version"),
    };
    let text = std::str::from_utf8(&bytes[start..start + length])?;
    let descr = extract(text, "'descr':")
        .ok_or_else(|| anyhow!("Missing descr"))?
        .trim()
        .trim_matches(|c| c == '\'' || c == '"')
        .to_string();
    let fortran = extract(text, "'fortran_order':")
        .map(|v| v.trim().starts_with("True"))
        .unwrap_or(false);
    let shape_text = text
        .split("'shape':")
        .nth(1)
        .ok_or_else(|| anyhow!("Missing shape"))?;
    let open = shape_text.find('(').ok_or_else(|| anyhow!("Bad shape"))?;
    let close = shape_text.find(')').ok_or_else(|| anyhow!("Bad shape"))?;
    let shape = shape_text[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|e| anyhow!(e)))
        .collect::<Result<Vec<_>>>()?;
    Ok(Parsed {
        descr,
        fortran,
        shape,
        offset: start + length,
    })
}

fn extract<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let rest = text.split(key).nth(1)?;
    rest.split(',').next()
}

fn load_raw<T: NpyElement>(path: &Path) -> Result<(Vec<usize>, Vec<T>)> {
    let bytes = fs::read(path).with_context(|| format!("Cannot read {}", path.display()))?;
    let parsed = parse_header(&bytes)?;
    let accepted = match T::DESCR {
        "|b1" => matches!(parsed.descr.as_str(), "|b1" | "b1" | "?" | "|?"),
        "|i1" => matches!(parsed.descr.as_str(), "|i1" | "i1"),
        other => parsed.descr == other,
    };
    if !accepted {
        bail!(
            "{}: expected dtype {}, found {}",
            path.display(),
            T::DESCR,
            parsed.descr
        );
    }
    let count: usize = parsed.shape.iter().product();
    let data = &bytes[parsed.offset..];
    if data.len() < count * T::SIZE {
        bail!("{}: truncated array", path.display());
    }
    let mut values: Vec<T> = (0..count)
        .map(|i| T::read(&data[i * T::SIZE..(i + 1) * T::SIZE]))
        .collect();
    if parsed.fortran && parsed.shape.len() == 2 {
        let (rows, cols) = (parsed.shape[0], parsed.shape[1]);
        let mut c_order = vec![T::default(); count];
        for col in 0..cols {
            for row in 0..rows {
                c_order[row * cols + col] = values[col * rows + row];
            }
        }
        values = c_order;
    }
    Ok((parsed.shape, values))
}

pub fn load_vec<T: NpyElement>(path: &Path) -> Result<Vec<T>> {
    let (shape, values) = load_raw::<T>(path)?;
    if shape.len() != 1 {
        bail!("{}: expected a one-dimensional array", path.display());
    }
    Ok(values)
}

pub fn load_mat<T: NpyElement>(path: &Path) -> Result<Mat<T>> {
    let (shape, values) = load_raw::<T>(path)?;
    if shape.len() != 2 {
        bail!("Invalid array shape: {}", path.display());
    }
    Ok(Mat {
        rows: shape[0],
        cols: shape[1],
        data: values,
    })
}

/// Return the dimensionality recorded in an array header.
pub fn ndim(path: &Path) -> Result<usize> {
    let bytes = fs::read(path)?;
    Ok(parse_header(&bytes)?.shape.len())
}
