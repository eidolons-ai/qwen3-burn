use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::model::Qwen3Config;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as little-endian u32

/// GGUF tensor data types we support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub(crate) enum GgufDtype {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q8_0 = 8,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    BF16 = 30,
}

impl GgufDtype {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            8 => Some(Self::Q8_0),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Number of bytes per block for quantized types, or per element for float types.
    fn block_size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            // Q8_0: 32 elements per block, 2 bytes scale + 32 bytes quants = 34 bytes
            Self::Q8_0 => 34,
            // Q4_0: 32 elements per block, 2 bytes scale + 16 bytes packed nibbles = 18 bytes
            Self::Q4_0 => 18,
            // K-quant types: 256 elements per super-block
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
        }
    }

    /// Number of elements per block (1 for unquantized types).
    fn block_size_elements(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q8_0 | Self::Q4_0 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K => 256,
        }
    }
}

/// Compute the byte size of tensor data given element count and dtype.
pub(crate) fn compute_data_size(num_elements: usize, dtype: GgufDtype) -> usize {
    let block_elems = dtype.block_size_elements();
    let num_blocks = num_elements / block_elems;
    num_blocks * dtype.block_size_bytes()
}

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub(crate) enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Uint64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    /// Try to extract as u64, coercing integer types.
    pub(crate) fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(v) => Some(*v as u64),
            Self::Int8(v) if *v >= 0 => Some(*v as u64),
            Self::Uint16(v) => Some(*v as u64),
            Self::Int16(v) if *v >= 0 => Some(*v as u64),
            Self::Uint32(v) => Some(*v as u64),
            Self::Int32(v) if *v >= 0 => Some(*v as u64),
            Self::Uint64(v) => Some(*v),
            Self::Int64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    /// Try to extract as f64, coercing float types.
    pub(crate) fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float32(v) => Some(*v as f64),
            Self::Float64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as bool.
    pub(crate) fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to extract as string.
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v.as_str()),
            _ => None,
        }
    }
}

/// Information about a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub(crate) struct GgufTensorInfo {
    pub name: String,
    /// Dimensions in GGUF ordering (reversed from PyTorch convention).
    pub dims: Vec<usize>,
    pub dtype: GgufDtype,
    pub offset: u64,
    pub num_elements: usize,
    pub data_size: usize,
}

/// Parsed GGUF file header and metadata.
pub(crate) struct GgufFile {
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    pub tensor_data_offset: u64,
}

// --- Reader helpers ---

fn read_u8(r: &mut impl Read) -> std::io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8(r: &mut impl Read) -> std::io::Result<i8> {
    Ok(read_u8(r)? as i8)
}

fn read_u16(r: &mut impl Read) -> std::io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16(r: &mut impl Read) -> std::io::Result<i16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> std::io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(r: &mut impl Read) -> std::io::Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32(r: &mut impl Read) -> std::io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> std::io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_bool(r: &mut impl Read) -> std::io::Result<bool> {
    Ok(read_u8(r)? != 0)
}

fn read_gguf_string(r: &mut impl Read) -> std::io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn read_metadata_value(r: &mut impl Read) -> std::io::Result<MetadataValue> {
    let type_tag = read_u32(r)?;
    read_metadata_value_typed(r, type_tag)
}

fn read_metadata_value_typed(r: &mut impl Read, type_tag: u32) -> std::io::Result<MetadataValue> {
    match type_tag {
        0 => Ok(MetadataValue::Uint8(read_u8(r)?)),
        1 => Ok(MetadataValue::Int8(read_i8(r)?)),
        2 => Ok(MetadataValue::Uint16(read_u16(r)?)),
        3 => Ok(MetadataValue::Int16(read_i16(r)?)),
        4 => Ok(MetadataValue::Uint32(read_u32(r)?)),
        5 => Ok(MetadataValue::Int32(read_i32(r)?)),
        6 => Ok(MetadataValue::Float32(read_f32(r)?)),
        7 => Ok(MetadataValue::Bool(read_bool(r)?)),
        8 => Ok(MetadataValue::String(read_gguf_string(r)?)),
        9 => {
            // Array: element type tag + count + elements
            let elem_type = read_u32(r)?;
            let count = read_u64(r)? as usize;
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(read_metadata_value_typed(r, elem_type)?);
            }
            Ok(MetadataValue::Array(values))
        }
        10 => Ok(MetadataValue::Uint64(read_u64(r)?)),
        11 => Ok(MetadataValue::Int64(read_i64(r)?)),
        12 => Ok(MetadataValue::Float64(read_f64(r)?)),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unknown GGUF metadata type tag: {}", type_tag),
        )),
    }
}

/// Align offset to the given alignment boundary.
fn align_offset(offset: u64, alignment: u64) -> u64 {
    let remainder = offset % alignment;
    if remainder == 0 {
        offset
    } else {
        offset + (alignment - remainder)
    }
}

impl GgufFile {
    /// Open and parse a GGUF file, returning the parsed metadata and a file handle
    /// positioned for tensor data reading.
    pub fn open(path: impl AsRef<Path>) -> Result<(Self, File), Box<dyn std::error::Error>> {
        let mut file = File::open(path.as_ref())?;

        // Read header
        let magic = read_u32(&mut file)?;
        if magic != GGUF_MAGIC {
            return Err(format!(
                "Invalid GGUF magic: expected 0x{:08X}, got 0x{:08X}",
                GGUF_MAGIC, magic
            )
            .into());
        }

        let version = read_u32(&mut file)?;
        if !(2..=3).contains(&version) {
            return Err(format!("Unsupported GGUF version: {} (expected 2 or 3)", version).into());
        }

        let tensor_count = read_u64(&mut file)? as usize;
        let kv_count = read_u64(&mut file)? as usize;

        // Read metadata key-value pairs
        let mut metadata = HashMap::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = read_gguf_string(&mut file)?;
            let value = read_metadata_value(&mut file)?;
            metadata.insert(key, value);
        }

        // Read tensor info entries
        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = read_gguf_string(&mut file)?;
            let n_dims = read_u32(&mut file)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut file)? as usize);
            }
            let dtype_raw = read_u32(&mut file)?;
            let dtype = GgufDtype::from_u32(dtype_raw).ok_or_else(|| {
                format!(
                    "Unsupported GGUF dtype {} for tensor '{}'. Supported: F32(0), F16(1), Q4_0(2), Q8_0(8), Q2_K(10), Q3_K(11), Q4_K(12), Q5_K(13), Q6_K(14), BF16(30)",
                    dtype_raw, name
                )
            })?;
            let offset = read_u64(&mut file)?;

            let num_elements = if dims.is_empty() {
                1
            } else {
                dims.iter().product()
            };
            let data_size = compute_data_size(num_elements, dtype);

            tensors.insert(
                name.clone(),
                GgufTensorInfo {
                    name,
                    dims,
                    dtype,
                    offset,
                    num_elements,
                    data_size,
                },
            );
        }

        // The tensor data starts after header+metadata+tensor_info, aligned to `alignment` bytes.
        // Default alignment is 32 bytes; can be overridden in metadata.
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32);

        let current_pos = file.stream_position()?;
        let tensor_data_offset = align_offset(current_pos, alignment);

        Ok((
            Self {
                metadata,
                tensors,
                tensor_data_offset,
            },
            file,
        ))
    }

    /// Query the dtype of a tensor by name.
    pub fn tensor_dtype(&self, name: &str) -> Option<GgufDtype> {
        self.tensors.get(name).map(|t| t.dtype)
    }

    /// Read and dequantize a tensor's data to f32.
    pub fn read_tensor_data(
        &self,
        file: &mut File,
        tensor: &GgufTensorInfo,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let abs_offset = self.tensor_data_offset + tensor.offset;
        file.seek(SeekFrom::Start(abs_offset))?;

        let mut raw = vec![0u8; tensor.data_size];
        file.read_exact(&mut raw)?;

        let data = match tensor.dtype {
            GgufDtype::F32 => dequantize_f32(&raw),
            GgufDtype::F16 => dequantize_f16(&raw),
            GgufDtype::BF16 => dequantize_bf16(&raw),
            GgufDtype::Q8_0 => dequantize_q8_0(&raw),
            GgufDtype::Q4_0 => dequantize_q4_0(&raw),
            GgufDtype::Q2_K => dequantize_q2_k(&raw),
            GgufDtype::Q3_K => dequantize_q3_k(&raw),
            GgufDtype::Q4_K => dequantize_q4_k(&raw),
            GgufDtype::Q5_K => dequantize_q5_k(&raw),
            GgufDtype::Q6_K => dequantize_q6_k(&raw),
        };

        Ok(data)
    }
}

// --- Dequantization functions ---

pub(crate) fn dequantize_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub(crate) fn dequantize_f16(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

pub(crate) fn dequantize_bf16(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// Dequantize Q8_0: block_size=32, per block: f16 scale (2 bytes) + 32 int8 quants (32 bytes) = 34 bytes.
pub(crate) fn dequantize_q8_0(data: &[u8]) -> Vec<f32> {
    let block_bytes = 34;
    let num_blocks = data.len() / block_bytes;
    let mut output = Vec::with_capacity(num_blocks * 32);

    for block in data.chunks_exact(block_bytes) {
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        for &q in &block[2..34] {
            output.push(scale * (q as i8) as f32);
        }
    }

    output
}

/// Dequantize Q4_0: block_size=32, per block: f16 scale (2 bytes) + 16 packed nibble bytes (16 bytes) = 18 bytes.
/// Low nibble of byte[j] → value[j], high nibble → value[j+16], centered at 8.
pub(crate) fn dequantize_q4_0(data: &[u8]) -> Vec<f32> {
    let block_bytes = 18;
    let num_blocks = data.len() / block_bytes;
    let mut output = Vec::with_capacity(num_blocks * 32);

    for block in data.chunks_exact(block_bytes) {
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let nibbles = &block[2..18];

        for &byte in nibbles.iter().take(16) {
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            output.push(scale * lo as f32);
            output.push(scale * hi as f32);
        }
    }

    output
}

// --- K-quant dequantization ---

/// Extract (scale, min) for a sub-block from the packed 12-byte scales array used by Q4_K and Q5_K.
fn get_scale_min_k4(scales: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        let d = scales[j] & 63;
        let m = scales[j + 4] & 63;
        (d, m)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize Q6_K: 256-element super-blocks, symmetric (no min), signed i8 scales.
///
/// Block layout (210 bytes):
///   ql[128]: lower 4 bits of 6-bit quants (interleaved across 128-element halves)
///   qh[64]:  upper 2 bits of 6-bit quants (4 values per byte, interleaved)
///   scales[16]: signed i8 sub-block scales
///   d: f16 super-block scale
///
/// Element ordering follows llama.cpp: each 128-element half uses 64 ql bytes and
/// 32 qh bytes. For l=0..31, four output positions are computed simultaneously:
///   y[l+0]  from ql[l] low nibble,    qh[l] bits [0:1]
///   y[l+32] from ql[l+32] low nibble, qh[l] bits [2:3]
///   y[l+64] from ql[l] high nibble,   qh[l] bits [4:5]
///   y[l+96] from ql[l+32] high nibble, qh[l] bits [6:7]
pub(crate) fn dequantize_q6_k(data: &[u8]) -> Vec<f32> {
    let block_bytes = 210;
    let num_blocks = data.len() / block_bytes;
    let mut output = Vec::with_capacity(num_blocks * 256);

    for block in data.chunks_exact(block_bytes) {
        let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();

        // Process two 128-element halves
        let mut ql_off: usize = 0; // into block[0..128]
        let mut qh_off: usize = 128; // into block[128..192]
        let mut sc_off: usize = 192; // into block[192..208]

        for _half in 0..2 {
            let mut y = [0.0f32; 128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = (((block[ql_off + l] & 0xF) | ((block[qh_off + l] & 3) << 4)) as i32) - 32;
                let q2 = (((block[ql_off + l + 32] & 0xF) | (((block[qh_off + l] >> 2) & 3) << 4))
                    as i32)
                    - 32;
                let q3 = (((block[ql_off + l] >> 4) | (((block[qh_off + l] >> 4) & 3) << 4))
                    as i32)
                    - 32;
                let q4 = (((block[ql_off + l + 32] >> 4) | (((block[qh_off + l] >> 6) & 3) << 4))
                    as i32)
                    - 32;
                y[l] = d * (block[sc_off + is] as i8 as f32) * q1 as f32;
                y[l + 32] = d * (block[sc_off + is + 2] as i8 as f32) * q2 as f32;
                y[l + 64] = d * (block[sc_off + is + 4] as i8 as f32) * q3 as f32;
                y[l + 96] = d * (block[sc_off + is + 6] as i8 as f32) * q4 as f32;
            }
            output.extend_from_slice(&y);
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }

    output
}

/// Dequantize Q4_K: 256-element super-blocks with asymmetric quantization (scale + min).
///
/// Block layout (144 bytes):
///   d: f16 super-block scale
///   dmin: f16 super-block min scale
///   scales[12]: packed sub-block scales and mins (via get_scale_min_k4)
///   qs[128]: packed 4-bit quants (2 per byte)
///
/// Element ordering: 4 chunks of 64. Each chunk reads 32 qs bytes — first 32 low
/// nibbles (scale is), then 32 high nibbles (scale is+1).
pub(crate) fn dequantize_q4_k(data: &[u8]) -> Vec<f32> {
    let block_bytes = 144;
    let num_blocks = data.len() / block_bytes;
    let mut output = Vec::with_capacity(num_blocks * 256);

    for block in data.chunks_exact(block_bytes) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qs = &block[16..144];

        let mut is = 0;
        let mut q_off = 0;
        for _j in 0..4 {
            let (sc1, m1) = get_scale_min_k4(scales, is);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(scales, is + 1);
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;
            for l in 0..32 {
                output.push(d1 * (qs[q_off + l] & 0xF) as f32 - m1);
            }
            for l in 0..32 {
                output.push(d2 * (qs[q_off + l] >> 4) as f32 - m2);
            }
            q_off += 32;
            is += 2;
        }
    }

    output
}

/// Dequantize Q5_K: 256-element super-blocks with 5-bit quants (scale + min).
///
/// Block layout (176 bytes):
///   d: f16 super-block scale
///   dmin: f16 super-block min scale
///   scales[12]: packed sub-block scales and mins (via get_scale_min_k4)
///   qh[32]: high bits — each byte provides bit 4 for 8 values across chunks
///   qs[128]: packed 4-bit quants (lower 4 bits, 2 per byte)
///
/// Element ordering: 4 chunks of 64. Each chunk reads 32 ql bytes — first 32 low
/// nibbles + high bits (scale is), then 32 high nibbles + high bits (scale is+1).
/// qh bit position advances by 2 per chunk (u1 and u2 shift left by 2).
pub(crate) fn dequantize_q5_k(data: &[u8]) -> Vec<f32> {
    let block_bytes = 176;
    let num_blocks = data.len() / block_bytes;
    let mut output = Vec::with_capacity(num_blocks * 256);

    for block in data.chunks_exact(block_bytes) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qh = &block[16..48];
        let ql = &block[48..176];

        let mut is = 0;
        let mut ql_off = 0;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _j in 0..4 {
            let (sc1, m1) = get_scale_min_k4(scales, is);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(scales, is + 1);
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;
            for l in 0..32 {
                let high: u8 = if qh[l] & u1 != 0 { 16 } else { 0 };
                output.push(d1 * ((ql[ql_off + l] & 0xF) + high) as f32 - m1);
            }
            for l in 0..32 {
                let high: u8 = if qh[l] & u2 != 0 { 16 } else { 0 };
                output.push(d2 * ((ql[ql_off + l] >> 4) + high) as f32 - m2);
            }
            ql_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    output
}

/// Dequantize Q2_K: 256-element super-blocks with 2-bit quants.
///
/// Block layout (84 bytes):
///   scales[16]: packed scale (low nibble) and min (high nibble) per sub-block
///   qs[64]: packed 2-bit quants (4 per byte)
///   d: f16 super-block scale
///   dmin: f16 super-block min scale
///
/// Element ordering: two 128-element halves. Each half uses 32 qs bytes at 4
/// bit shifts (0,2,4,6). For each shift, two 16-element sub-blocks are produced
/// from q[0..16] and q[16..32], each with its own scale byte.
pub(crate) fn dequantize_q2_k(data: &[u8]) -> Vec<f32> {
    let block_bytes = 84;
    let num_blocks = data.len() / block_bytes;
    let mut output = Vec::with_capacity(num_blocks * 256);

    for block in data.chunks_exact(block_bytes) {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = half::f16::from_le_bytes([block[80], block[81]]).to_f32();
        let dmin = half::f16::from_le_bytes([block[82], block[83]]).to_f32();

        let mut q_off: usize = 0;
        let mut is: usize = 0;
        for _half in 0..2 {
            let mut shift: u32 = 0;
            for _j in 0..4 {
                let sc_byte = scales[is];
                let dl = d * (sc_byte & 0xF) as f32;
                let ml = dmin * (sc_byte >> 4) as f32;
                is += 1;
                for l in 0..16 {
                    output.push(dl * ((qs[q_off + l] >> shift) & 3) as f32 - ml);
                }
                let sc_byte = scales[is];
                let dl = d * (sc_byte & 0xF) as f32;
                let ml = dmin * (sc_byte >> 4) as f32;
                is += 1;
                for l in 0..16 {
                    output.push(dl * ((qs[q_off + l + 16] >> shift) & 3) as f32 - ml);
                }
                shift += 2;
            }
            q_off += 32;
        }
    }

    output
}

/// Dequantize Q3_K: 256-element super-blocks with 3-bit quants.
///
/// Block layout (110 bytes):
///   hmask[32]: high bit mask (bit 2 of each 3-bit quant)
///   qs[64]: packed 2-bit quants (lower 2 bits, 4 per byte)
///   scales[12]: packed 6-bit signed scales for 16 sub-blocks
///   d: f16 super-block scale
///
/// Element ordering matches Q2_K: two 128-element halves, each using 32 qs bytes
/// at 4 bit shifts. hmask bit position `m` advances once per pair of sub-blocks
/// (starting at bit 0, incrementing through bit 7).
pub(crate) fn dequantize_q3_k(data: &[u8]) -> Vec<f32> {
    let block_bytes = 110;
    let num_blocks = data.len() / block_bytes;
    let mut output = Vec::with_capacity(num_blocks * 256);

    let kmask1: u32 = 0x03030303;
    let kmask2: u32 = 0x0F0F0F0F;

    for block in data.chunks_exact(block_bytes) {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales_raw = &block[96..108];
        let d = half::f16::from_le_bytes([block[108], block[109]]).to_f32();

        // Unpack 12 bytes into 16 six-bit scales via u32 bitmask manipulation.
        // Matches llama.cpp: aux[2],aux[3] computed first (from original a[0],a[1]),
        // then aux[0],aux[1] overwrite. We use separate a[] and aux[] arrays.
        let a0 = u32::from_le_bytes([scales_raw[0], scales_raw[1], scales_raw[2], scales_raw[3]]);
        let a1 = u32::from_le_bytes([scales_raw[4], scales_raw[5], scales_raw[6], scales_raw[7]]);
        let tmp =
            u32::from_le_bytes([scales_raw[8], scales_raw[9], scales_raw[10], scales_raw[11]]);

        let mut aux = [0u32; 4];
        aux[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (a0 & kmask2) | ((tmp & kmask1) << 4);
        aux[1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);

        // Convert 4 u32s to 16 scale bytes
        let mut scales = [0u8; 16];
        for i in 0..4 {
            let bytes = aux[i].to_le_bytes();
            scales[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }

        let mut q_off: usize = 0;
        let mut is: usize = 0;
        let mut m: u8 = 1;
        for _half in 0..2 {
            let mut shift: u32 = 0;
            for _j in 0..4 {
                let dl = d * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16usize {
                    let q_low = ((qs[q_off + l] >> shift) & 3) as i32;
                    let hm_sub = if hmask[l] & m != 0 { 0 } else { 4 };
                    output.push(dl * (q_low - hm_sub) as f32);
                }
                let dl = d * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16usize {
                    let q_low = ((qs[q_off + l + 16] >> shift) & 3) as i32;
                    let hm_sub = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                    output.push(dl * (q_low - hm_sub) as f32);
                }
                shift += 2;
                m <<= 1;
            }
            q_off += 32;
        }
    }

    output
}

// --- Tensor name mapping ---

/// Map a GGUF tensor name to the corresponding HuggingFace name.
/// Returns None for unknown tensor names.
pub(crate) fn map_tensor_name(gguf_name: &str) -> Option<String> {
    // Global tensors
    match gguf_name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".to_string()),
        "output_norm.weight" => return Some("model.norm.weight".to_string()),
        "output.weight" => return Some("lm_head.weight".to_string()),
        _ => {}
    }

    // Block-level tensors: blk.{i}.xxx
    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        let dot_pos = rest.find('.')?;
        let layer_idx = &rest[..dot_pos];
        let suffix = &rest[dot_pos + 1..];

        let hf_suffix = match suffix {
            "attn_norm.weight" => "input_layernorm.weight",
            "attn_q.weight" => "self_attn.q_proj.weight",
            "attn_k.weight" => "self_attn.k_proj.weight",
            "attn_v.weight" => "self_attn.v_proj.weight",
            "attn_output.weight" => "self_attn.o_proj.weight",
            "attn_q_norm.weight" => "self_attn.q_norm.weight",
            "attn_k_norm.weight" => "self_attn.k_norm.weight",
            "ffn_norm.weight" => "post_attention_layernorm.weight",
            "ffn_gate.weight" => "mlp.gate_proj.weight",
            "ffn_up.weight" => "mlp.up_proj.weight",
            "ffn_down.weight" => "mlp.down_proj.weight",
            // MoE tensors
            "ffn_gate_inp.weight" => "mlp.router.weight",
            "ffn_gate_exps.weight" => "mlp.experts.gate_proj",
            "ffn_up_exps.weight" => "mlp.experts.up_proj",
            "ffn_down_exps.weight" => "mlp.experts.down_proj",
            _ => return None,
        };

        return Some(format!("model.layers.{}.{}", layer_idx, hf_suffix));
    }

    None
}

/// Extract Qwen3Config from GGUF metadata.
///
/// The architecture prefix is read from `general.architecture` (typically "qwen3"
/// for official Qwen3 GGUFs, or "qwen2" for older conversions).
pub(crate) fn extract_config(gguf: &GgufFile) -> Result<Qwen3Config, Box<dyn std::error::Error>> {
    let md = &gguf.metadata;

    // Detect architecture prefix from metadata (e.g. "qwen3" or "qwen2")
    let arch = md
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("qwen2")
        .to_string();

    let get_u64 = |suffix: &str| -> Option<u64> {
        let key = format!("{}.{}", arch, suffix);
        md.get(&key).and_then(|v| v.as_u64())
    };
    let get_f64 = |suffix: &str| -> Option<f64> {
        let key = format!("{}.{}", arch, suffix);
        md.get(&key).and_then(|v| v.as_f64())
    };

    let hidden_size = get_u64("embedding_length")
        .ok_or_else(|| format!("Missing {}.embedding_length in GGUF metadata", arch))?
        as usize;
    let num_hidden_layers = get_u64("block_count")
        .ok_or_else(|| format!("Missing {}.block_count in GGUF metadata", arch))?
        as usize;
    let num_attention_heads = get_u64("attention.head_count")
        .ok_or_else(|| format!("Missing {}.attention.head_count in GGUF metadata", arch))?
        as usize;
    let num_key_value_heads = get_u64("attention.head_count_kv")
        .ok_or_else(|| format!("Missing {}.attention.head_count_kv in GGUF metadata", arch))?
        as usize;
    let intermediate_size = get_u64("feed_forward_length")
        .ok_or_else(|| format!("Missing {}.feed_forward_length in GGUF metadata", arch))?
        as usize;

    let rms_norm_eps = get_f64("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
    let rope_theta = get_f64("rope.freq_base").unwrap_or(1_000_000.0);
    let max_position_embeddings = get_u64("context_length").unwrap_or(40960) as usize;

    // Vocab size: inferred from embedding tensor shape
    let vocab_size = gguf
        .tensors
        .get("token_embd.weight")
        .map(|t| {
            // GGUF dims are reversed; for embedding [hidden, vocab] -> reversed = [vocab, hidden]
            if t.dims.len() == 2 {
                t.dims[1]
            } else {
                t.num_elements / hidden_size
            }
        })
        .ok_or("Missing token_embd.weight tensor to determine vocab_size")?;

    // tie_word_embeddings: true if no output.weight tensor
    let tie_word_embeddings = !gguf.tensors.contains_key("output.weight");

    // head_dim: default 128 for Qwen3
    let head_dim = get_u64("attention.key_length").unwrap_or(128) as usize;

    // MoE fields
    let num_experts = get_u64("expert_count").map(|v| v as usize);
    let num_experts_per_tok = get_u64("expert_used_count").map(|v| v as usize);

    // For MoE models, determine moe_intermediate_size from expert tensor shapes
    let moe_intermediate_size = if num_experts.is_some() {
        // Try to get from ffn_gate_exps tensor (first layer that has it)
        let mut moe_inter = None;
        for (name, info) in &gguf.tensors {
            if name.ends_with("ffn_gate_exps.weight") {
                // GGUF dims reversed: [hidden, moe_intermediate, num_experts]
                // -> PyTorch: [num_experts, moe_intermediate, hidden]
                if info.dims.len() == 3 {
                    moe_inter = Some(info.dims[1]);
                }
                break;
            }
        }
        moe_inter
    } else {
        None
    };

    let decoder_sparse_step = if num_experts.is_some() { Some(1) } else { None };

    let norm_topk_prob = if num_experts.is_some() {
        Some(true)
    } else {
        None
    };

    let mlp_only_layers = if num_experts.is_some() {
        Some(vec![])
    } else {
        None
    };

    Ok(Qwen3Config {
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        intermediate_size,
        vocab_size,
        max_position_embeddings,
        rms_norm_eps,
        rope_theta,
        head_dim,
        tie_word_embeddings,
        num_experts,
        num_experts_per_tok,
        moe_intermediate_size,
        decoder_sparse_step,
        norm_topk_prob,
        mlp_only_layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Dequantization tests ---

    #[test]
    fn dequantize_f32_known_values() {
        let val = 42.5f32;
        let bytes = val.to_le_bytes();
        let result = dequantize_f32(&bytes);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 42.5).abs() < 1e-6);
    }

    #[test]
    fn dequantize_f32_multiple() {
        let mut bytes = Vec::new();
        for v in [1.0f32, -2.0, 0.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let result = dequantize_f32(&bytes);
        assert_eq!(result, vec![1.0, -2.0, 0.0]);
    }

    #[test]
    fn dequantize_f32_empty() {
        let result = dequantize_f32(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn dequantize_f16_known_values() {
        let val = half::f16::from_f32(1.5);
        let bytes = val.to_le_bytes();
        let result = dequantize_f16(&bytes);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.5).abs() < 1e-3);
    }

    #[test]
    fn dequantize_f16_empty() {
        let result = dequantize_f16(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn dequantize_bf16_known_values() {
        let val = half::bf16::from_f32(2.0);
        let bytes = val.to_le_bytes();
        let result = dequantize_bf16(&bytes);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 2.0).abs() < 1e-2);
    }

    #[test]
    fn dequantize_bf16_empty() {
        let result = dequantize_bf16(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn dequantize_q8_0_known_values() {
        // Build one Q8_0 block: scale=2.0 (f16), 32 quants
        let scale = half::f16::from_f32(2.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale.to_le_bytes());
        // Quants: 1, -1, 0, 2, ... (fill 32 values)
        let quants: Vec<i8> = (0..32).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        for &q in &quants {
            block.push(q as u8);
        }
        assert_eq!(block.len(), 34);

        let result = dequantize_q8_0(&block);
        assert_eq!(result.len(), 32);
        // val = scale * quant = 2.0 * 1 = 2.0, 2.0 * -1 = -2.0, ...
        assert!((result[0] - 2.0).abs() < 1e-2);
        assert!((result[1] - (-2.0)).abs() < 1e-2);
    }

    #[test]
    fn dequantize_q8_0_zero_scale() {
        let scale = half::f16::from_f32(0.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale.to_le_bytes());
        block.extend(std::iter::repeat_n(127u8, 32)); // max positive i8
        let result = dequantize_q8_0(&block);
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequantize_q8_0_empty() {
        let result = dequantize_q8_0(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn dequantize_q4_0_known_values() {
        // Build one Q4_0 block: scale=1.0 (f16), 16 nibble bytes
        let scale = half::f16::from_f32(1.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale.to_le_bytes());
        // Nibble byte where lo=8 (value=0), hi=9 (value=1)
        // lo nibble = 8 -> 8-8 = 0, hi nibble = 9 -> 9-8 = 1
        for _ in 0..16 {
            let byte = (9 << 4) | 8; // lo=8, hi=9
            block.push(byte);
        }
        assert_eq!(block.len(), 18);

        let result = dequantize_q4_0(&block);
        assert_eq!(result.len(), 32);
        // All lo values: 0.0, all hi values: 1.0
        // Output order: lo[0], hi[0], lo[1], hi[1], ...
        for i in 0..16 {
            assert!(
                (result[i * 2] - 0.0).abs() < 1e-3,
                "lo[{}] = {}, expected 0.0",
                i,
                result[i * 2]
            );
            assert!(
                (result[i * 2 + 1] - 1.0).abs() < 1e-3,
                "hi[{}] = {}, expected 1.0",
                i,
                result[i * 2 + 1]
            );
        }
    }

    #[test]
    fn dequantize_q4_0_zero_scale() {
        let scale = half::f16::from_f32(0.0);
        let mut block = Vec::new();
        block.extend_from_slice(&scale.to_le_bytes());
        block.extend(std::iter::repeat_n(0xFFu8, 16)); // both nibbles = 15, but scale=0
        let result = dequantize_q4_0(&block);
        assert_eq!(result.len(), 32);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequantize_q4_0_negative_values() {
        // nibble=0 -> value = 0-8 = -8
        let scale = half::f16::from_f32(0.5);
        let mut block = Vec::new();
        block.extend_from_slice(&scale.to_le_bytes());
        block.extend(std::iter::repeat_n(0x00u8, 16)); // lo=0, hi=0 -> both = -8 * 0.5 = -4.0
        let result = dequantize_q4_0(&block);
        for &v in &result {
            assert!((v - (-4.0)).abs() < 1e-2, "expected -4.0 got {}", v);
        }
    }

    #[test]
    fn dequantize_q4_0_empty() {
        let result = dequantize_q4_0(&[]);
        assert!(result.is_empty());
    }

    // --- Name mapping tests ---

    #[test]
    fn map_global_tensor_names() {
        assert_eq!(
            map_tensor_name("token_embd.weight"),
            Some("model.embed_tokens.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("output_norm.weight"),
            Some("model.norm.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("output.weight"),
            Some("lm_head.weight".to_string())
        );
    }

    #[test]
    fn map_attention_tensor_names() {
        assert_eq!(
            map_tensor_name("blk.0.attn_q.weight"),
            Some("model.layers.0.self_attn.q_proj.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("blk.5.attn_k.weight"),
            Some("model.layers.5.self_attn.k_proj.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("blk.10.attn_v.weight"),
            Some("model.layers.10.self_attn.v_proj.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("blk.0.attn_output.weight"),
            Some("model.layers.0.self_attn.o_proj.weight".to_string())
        );
    }

    #[test]
    fn map_qk_norm_tensor_names() {
        assert_eq!(
            map_tensor_name("blk.3.attn_q_norm.weight"),
            Some("model.layers.3.self_attn.q_norm.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("blk.3.attn_k_norm.weight"),
            Some("model.layers.3.self_attn.k_norm.weight".to_string())
        );
    }

    #[test]
    fn map_ffn_dense_tensor_names() {
        assert_eq!(
            map_tensor_name("blk.2.ffn_gate.weight"),
            Some("model.layers.2.mlp.gate_proj.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("blk.2.ffn_up.weight"),
            Some("model.layers.2.mlp.up_proj.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("blk.2.ffn_down.weight"),
            Some("model.layers.2.mlp.down_proj.weight".to_string())
        );
    }

    #[test]
    fn map_ffn_moe_tensor_names() {
        assert_eq!(
            map_tensor_name("blk.0.ffn_gate_inp.weight"),
            Some("model.layers.0.mlp.router.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("blk.1.ffn_gate_exps.weight"),
            Some("model.layers.1.mlp.experts.gate_proj".to_string())
        );
        assert_eq!(
            map_tensor_name("blk.1.ffn_up_exps.weight"),
            Some("model.layers.1.mlp.experts.up_proj".to_string())
        );
        assert_eq!(
            map_tensor_name("blk.1.ffn_down_exps.weight"),
            Some("model.layers.1.mlp.experts.down_proj".to_string())
        );
    }

    #[test]
    fn map_layer_norm_tensor_names() {
        assert_eq!(
            map_tensor_name("blk.7.attn_norm.weight"),
            Some("model.layers.7.input_layernorm.weight".to_string())
        );
        assert_eq!(
            map_tensor_name("blk.7.ffn_norm.weight"),
            Some("model.layers.7.post_attention_layernorm.weight".to_string())
        );
    }

    #[test]
    fn map_unknown_returns_none() {
        assert_eq!(map_tensor_name("unknown_tensor"), None);
        assert_eq!(map_tensor_name("blk.0.unknown_suffix.weight"), None);
        assert_eq!(map_tensor_name(""), None);
    }

    // --- MetadataValue accessor tests ---

    #[test]
    fn metadata_as_u64_coercion() {
        assert_eq!(MetadataValue::Uint8(42).as_u64(), Some(42));
        assert_eq!(MetadataValue::Int8(10).as_u64(), Some(10));
        assert_eq!(MetadataValue::Int8(-1).as_u64(), None);
        assert_eq!(MetadataValue::Uint16(1000).as_u64(), Some(1000));
        assert_eq!(MetadataValue::Int32(999).as_u64(), Some(999));
        assert_eq!(MetadataValue::Int32(-5).as_u64(), None);
        assert_eq!(MetadataValue::Uint64(123456).as_u64(), Some(123456));
        assert_eq!(MetadataValue::Float32(1.0).as_u64(), None);
    }

    #[test]
    fn metadata_as_f64_coercion() {
        assert_eq!(MetadataValue::Float32(1.5).as_f64(), Some(1.5f32 as f64));
        assert_eq!(MetadataValue::Float64(99.5).as_f64(), Some(99.5));
        assert_eq!(MetadataValue::Uint32(1).as_f64(), None);
    }

    #[test]
    fn metadata_as_bool() {
        assert_eq!(MetadataValue::Bool(true).as_bool(), Some(true));
        assert_eq!(MetadataValue::Bool(false).as_bool(), Some(false));
        assert_eq!(MetadataValue::Uint8(1).as_bool(), None);
    }

    #[test]
    fn metadata_as_str() {
        let val = MetadataValue::String("hello".to_string());
        assert_eq!(val.as_str(), Some("hello"));
        assert_eq!(MetadataValue::Uint32(0).as_str(), None);
    }

    // --- Data size tests ---

    #[test]
    fn compute_data_size_f32() {
        assert_eq!(compute_data_size(100, GgufDtype::F32), 400);
        assert_eq!(compute_data_size(1, GgufDtype::F32), 4);
        assert_eq!(compute_data_size(0, GgufDtype::F32), 0);
    }

    #[test]
    fn compute_data_size_q8_0() {
        // 32 elements per block, 34 bytes per block
        assert_eq!(compute_data_size(32, GgufDtype::Q8_0), 34);
        assert_eq!(compute_data_size(64, GgufDtype::Q8_0), 68);
        assert_eq!(compute_data_size(0, GgufDtype::Q8_0), 0);
    }

    #[test]
    fn compute_data_size_q4_0() {
        // 32 elements per block, 18 bytes per block
        assert_eq!(compute_data_size(32, GgufDtype::Q4_0), 18);
        assert_eq!(compute_data_size(64, GgufDtype::Q4_0), 36);
        assert_eq!(compute_data_size(0, GgufDtype::Q4_0), 0);
    }

    // --- Alignment tests ---

    #[test]
    fn align_offset_already_aligned() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(64, 32), 64);
    }

    #[test]
    fn align_offset_unaligned() {
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(31, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
        assert_eq!(align_offset(100, 32), 128);
    }

    // --- Config extraction test ---

    #[test]
    fn extract_config_from_synthetic_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "qwen2.embedding_length".to_string(),
            MetadataValue::Uint32(1024),
        );
        metadata.insert("qwen2.block_count".to_string(), MetadataValue::Uint32(28));
        metadata.insert(
            "qwen2.attention.head_count".to_string(),
            MetadataValue::Uint32(16),
        );
        metadata.insert(
            "qwen2.attention.head_count_kv".to_string(),
            MetadataValue::Uint32(8),
        );
        metadata.insert(
            "qwen2.feed_forward_length".to_string(),
            MetadataValue::Uint32(3072),
        );
        metadata.insert(
            "qwen2.attention.layer_norm_rms_epsilon".to_string(),
            MetadataValue::Float32(1e-6),
        );
        metadata.insert(
            "qwen2.rope.freq_base".to_string(),
            MetadataValue::Float32(1_000_000.0),
        );
        metadata.insert(
            "qwen2.context_length".to_string(),
            MetadataValue::Uint32(40960),
        );

        // Create embedding tensor info for vocab size inference
        // GGUF dims are reversed: [hidden, vocab] = [1024, 151936]
        let mut tensors = HashMap::new();
        tensors.insert(
            "token_embd.weight".to_string(),
            GgufTensorInfo {
                name: "token_embd.weight".to_string(),
                dims: vec![1024, 151936],
                dtype: GgufDtype::F16,
                offset: 0,
                num_elements: 1024 * 151936,
                data_size: 1024 * 151936 * 2,
            },
        );
        // No output.weight -> tie_word_embeddings = true

        let gguf = GgufFile {
            metadata,
            tensors,
            tensor_data_offset: 0,
        };

        let config = extract_config(&gguf).unwrap();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.intermediate_size, 3072);
        assert_eq!(config.vocab_size, 151936);
        assert!(config.tie_word_embeddings);
        assert!(!config.is_moe());
        assert!((config.rms_norm_eps - 1e-6f64).abs() < 1e-10);
        assert_eq!(config.rope_theta, 1_000_000.0);
        assert_eq!(config.head_dim, 128);
    }

    // --- GgufDtype tests ---

    #[test]
    fn gguf_dtype_from_u32_valid() {
        assert_eq!(GgufDtype::from_u32(0), Some(GgufDtype::F32));
        assert_eq!(GgufDtype::from_u32(1), Some(GgufDtype::F16));
        assert_eq!(GgufDtype::from_u32(2), Some(GgufDtype::Q4_0));
        assert_eq!(GgufDtype::from_u32(8), Some(GgufDtype::Q8_0));
        assert_eq!(GgufDtype::from_u32(30), Some(GgufDtype::BF16));
    }

    #[test]
    fn gguf_dtype_from_u32_k_quants() {
        assert_eq!(GgufDtype::from_u32(10), Some(GgufDtype::Q2_K));
        assert_eq!(GgufDtype::from_u32(11), Some(GgufDtype::Q3_K));
        assert_eq!(GgufDtype::from_u32(12), Some(GgufDtype::Q4_K));
        assert_eq!(GgufDtype::from_u32(13), Some(GgufDtype::Q5_K));
        assert_eq!(GgufDtype::from_u32(14), Some(GgufDtype::Q6_K));
    }

    #[test]
    fn gguf_dtype_from_u32_invalid() {
        assert_eq!(GgufDtype::from_u32(3), None);
        assert_eq!(GgufDtype::from_u32(99), None);
        assert_eq!(GgufDtype::from_u32(255), None);
    }

    #[test]
    fn compute_data_size_f16() {
        assert_eq!(compute_data_size(100, GgufDtype::F16), 200);
    }

    #[test]
    fn compute_data_size_bf16() {
        assert_eq!(compute_data_size(100, GgufDtype::BF16), 200);
    }

    #[test]
    fn compute_data_size_k_quants() {
        // All k-quants: 256 elements per block
        assert_eq!(compute_data_size(256, GgufDtype::Q2_K), 84);
        assert_eq!(compute_data_size(256, GgufDtype::Q3_K), 110);
        assert_eq!(compute_data_size(256, GgufDtype::Q4_K), 144);
        assert_eq!(compute_data_size(256, GgufDtype::Q5_K), 176);
        assert_eq!(compute_data_size(256, GgufDtype::Q6_K), 210);
        // Two blocks
        assert_eq!(compute_data_size(512, GgufDtype::Q4_K), 288);
        assert_eq!(compute_data_size(512, GgufDtype::Q6_K), 420);
    }

    // --- K-quant dequantization tests ---

    #[test]
    fn dequantize_q6_k_known_values() {
        // Build one Q6_K block (210 bytes): ql[128] | qh[64] | scales[16] | d(f16)
        let mut block = vec![0u8; 210];

        // Set d = 1.0
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[208] = d_bytes[0];
        block[209] = d_bytes[1];

        // Set scale[0] = 2 (signed i8)
        block[192] = 2u8;

        // Set first element: ql low nibble = 5, qh bits = 0 -> 6-bit val = 5, result = 1.0 * 2.0 * (5 - 32) = -54
        block[0] = 5; // ql[0] low nibble = 5, high nibble = 0

        let result = dequantize_q6_k(&block);
        assert_eq!(result.len(), 256);
        // Element 0: d * scale * (ql_lo | (qh << 4) - 32) = 1.0 * 2.0 * (5 - 32) = -54.0
        assert!((result[0] - (-54.0)).abs() < 1e-3, "got {}", result[0]);
    }

    #[test]
    fn dequantize_q6_k_zero_scale() {
        let block = vec![0u8; 210];
        // d = 0 (default), all values should be 0
        let result = dequantize_q6_k(&block);
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequantize_q6_k_empty() {
        let result = dequantize_q6_k(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn dequantize_q4_k_known_values() {
        // Build one Q4_K block (144 bytes): d(f16) | dmin(f16) | scales[12] | qs[128]
        let mut block = vec![0u8; 144];

        // d = 2.0, dmin = 1.0
        let d_bytes = half::f16::from_f32(2.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        let dmin_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[2] = dmin_bytes[0];
        block[3] = dmin_bytes[1];

        // Sub-block 0 (is=0): sc=3, m=1; Sub-block 1 (is=1): sc=2, m=1
        // get_scale_min_k4(scales, 0) → (scales[0] & 63, scales[4] & 63) = (3, 1)
        // get_scale_min_k4(scales, 1) → (scales[1] & 63, scales[5] & 63) = (2, 1)
        block[4] = 3; // scales[0] = 3
        block[5] = 2; // scales[1] = 2
        block[8] = 1; // scales[4] = 1
        block[9] = 1; // scales[5] = 1

        // qs[0]: low nibble = 5, high nibble = 7
        // Element 0 reads low nibble (sub-block 0): d * 3 * 5 - dmin * 1 = 6*5 - 1 = 29.0
        // Element 32 reads high nibble (sub-block 1): d * 2 * 7 - dmin * 1 = 4*7 - 1 = 27.0
        block[16] = (7 << 4) | 5;

        let result = dequantize_q4_k(&block);
        assert_eq!(result.len(), 256);
        // Element 0: low nibble with sub-block 0 scale
        assert!((result[0] - 29.0).abs() < 1e-2, "elem 0 got {}", result[0]);
        // Element 32: high nibble with sub-block 1 scale
        assert!(
            (result[32] - 27.0).abs() < 1e-2,
            "elem 32 got {}",
            result[32]
        );
    }

    #[test]
    fn dequantize_q4_k_zero_scale() {
        let block = vec![0u8; 144];
        // d = 0, dmin = 0 (default), all values should be 0
        let result = dequantize_q4_k(&block);
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequantize_q4_k_empty() {
        let result = dequantize_q4_k(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn dequantize_q5_k_known_values() {
        // Build one Q5_K block (176 bytes): d(f16) | dmin(f16) | scales[12] | qh[32] | qs[128]
        let mut block = vec![0u8; 176];

        // d = 1.0, dmin = 0.5
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = d_bytes[0];
        block[1] = d_bytes[1];
        let dmin_bytes = half::f16::from_f32(0.5).to_le_bytes();
        block[2] = dmin_bytes[0];
        block[3] = dmin_bytes[1];

        // Sub-block 0 (is=0): sc=4, m=2; Sub-block 1 (is=1): sc=3, m=1
        block[4] = 4; // scales[0]
        block[5] = 3; // scales[1]
        block[8] = 2; // scales[4]
        block[9] = 1; // scales[5]

        // ql[0] at offset 48: low nibble = 3, high nibble = 7
        block[48] = (7 << 4) | 3;

        // qh[0] at offset 16: bit 0 (u1=1 for chunk 0 low nibbles), bit 1 (u2=2 for chunk 0 high nibbles)
        // Set bit 0 → high bit for element 0 low nibble = 16
        // Set bit 1 → high bit for element 32 high nibble = 16
        block[16] = 0x03; // bits 0 and 1 set

        let result = dequantize_q5_k(&block);
        assert_eq!(result.len(), 256);
        // Element 0 (first chunk, low nibble): ql=3, qh bit 0 set → q = 3+16 = 19
        //   d * sc * q - dmin * m = 1.0 * 4 * 19 - 0.5 * 2 = 76 - 1 = 75.0
        assert!((result[0] - 75.0).abs() < 1e-2, "elem 0 got {}", result[0]);
        // Element 32 (first chunk, high nibble): ql=7, qh bit 1 set → q = 7+16 = 23
        //   d * sc * q - dmin * m = 1.0 * 3 * 23 - 0.5 * 1 = 69 - 0.5 = 68.5
        assert!(
            (result[32] - 68.5).abs() < 1e-2,
            "elem 32 got {}",
            result[32]
        );
    }

    #[test]
    fn dequantize_q5_k_zero_scale() {
        let block = vec![0u8; 176];
        let result = dequantize_q5_k(&block);
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequantize_q5_k_empty() {
        let result = dequantize_q5_k(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn dequantize_q2_k_known_values() {
        // Build one Q2_K block (84 bytes): scales[16] | qs[64] | d(f16) | dmin(f16)
        let mut block = vec![0u8; 84];

        // d = 3.0, dmin = 1.0
        let d_bytes = half::f16::from_f32(3.0).to_le_bytes();
        block[80] = d_bytes[0];
        block[81] = d_bytes[1];
        let dmin_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[82] = dmin_bytes[0];
        block[83] = dmin_bytes[1];

        // scales[0] = sc=5 (low nibble) | m=2 (high nibble) = 0x25
        block[0] = 0x25;

        // qs[0] = first 4 elements as 2-bit values: [1, 2, 3, 0] packed
        // bits: 00_11_10_01 = 0b00_11_10_01 = 0xE1  (wait, let me be more careful)
        // Element 0: bits [0:1] = 1  (0b01)
        // Element 1: bits [2:3] = 2  (0b10)
        // Element 2: bits [4:5] = 3  (0b11)
        // Element 3: bits [6:7] = 0  (0b00)
        // Packed: 0b_00_11_10_01 = 0x39
        block[16] = 0x39;

        let result = dequantize_q2_k(&block);
        assert_eq!(result.len(), 256);
        // In the corrected element ordering, each qs byte is read at 4 different bit shifts
        // for 4 different output positions separated by 32 elements.
        // qs[0] = 0x39 = 0b00_11_10_01:
        //   shift 0 → bits[0:1] = 1 → element 0  (sub-block 0, scales[0])
        //   shift 2 → bits[2:3] = 2 → element 32 (sub-block 2, scales[2])
        //   shift 4 → bits[4:5] = 3 → element 64 (sub-block 4, scales[4])
        //   shift 6 → bits[6:7] = 0 → element 96 (sub-block 6, scales[6])
        // Only scales[0] is non-zero (0x25: sc=5, m=2), so only element 0 has a non-trivial value.
        // Element 0: d * sc * 1 - dmin * m = 3.0 * 5 * 1 - 1.0 * 2 = 13.0
        assert!((result[0] - 13.0).abs() < 1e-2, "got {}", result[0]);
        // Element 1 reads qs[1] = 0 at shift 0: d * sc * 0 - dmin * m = -2.0
        assert!((result[1] - (-2.0)).abs() < 1e-2, "got {}", result[1]);
        // Elements at shifts 2,4,6 use scales[2,4,6] = 0, so dl=0, ml=0 → 0.0
        assert!((result[32]).abs() < 1e-2, "elem 32 got {}", result[32]);
        assert!((result[64]).abs() < 1e-2, "elem 64 got {}", result[64]);
        assert!((result[96]).abs() < 1e-2, "elem 96 got {}", result[96]);
    }

    #[test]
    fn dequantize_q2_k_zero_scale() {
        let block = vec![0u8; 84];
        let result = dequantize_q2_k(&block);
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequantize_q2_k_empty() {
        let result = dequantize_q2_k(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn dequantize_q3_k_known_values() {
        // Build one Q3_K block (110 bytes): hmask[32] | qs[64] | scales[12] | d(f16)
        let mut block = vec![0u8; 110];

        // d = 1.0
        let d_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[108] = d_bytes[0];
        block[109] = d_bytes[1];

        // Set all scales to a known value. Scales are packed as 12 bytes → 16 six-bit values.
        // Simplest: set all 12 bytes to 0 → all scales decode to some value, then subtract 32.
        // With all zeros, all utmp = 0, so all scales = 0 - 32 = -32.
        // Let's instead set scale bytes so that scale[0] = 32 (→ value 0 after subtracting 32)
        // For simplicity, just verify with all-zero scales (scale = -32 each).

        // qs[0] = first 4 elements with 2-bit values: [1, 0, 0, 0]
        // Element 0: bits [0:1] = 1
        block[32] = 0x01;

        // hmask: bit 0 set for element 0 (means do NOT subtract 4)
        block[0] = 0x01;

        let result = dequantize_q3_k(&block);
        assert_eq!(result.len(), 256);
        // Element 0: q_low=1, hmask bit set → q = 1 - 0 = 1
        //   d * scale * q = 1.0 * (-32) * 1 = -32.0
        assert!((result[0] - (-32.0)).abs() < 1e-2, "got {}", result[0]);
    }

    #[test]
    fn dequantize_q3_k_zero_scale() {
        let block = vec![0u8; 110];
        // d = 0 (default), all values should be 0
        let result = dequantize_q3_k(&block);
        assert_eq!(result.len(), 256);
        for &v in &result {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn dequantize_q3_k_empty() {
        let result = dequantize_q3_k(&[]);
        assert!(result.is_empty());
    }
}
