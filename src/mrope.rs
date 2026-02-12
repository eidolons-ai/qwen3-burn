use burn::prelude::*;

/// Multi-Resolution Rotary Position Embedding (mRoPE) for Qwen3-VL.
///
/// The config specifies `mrope_section = [24, 20, 20]` as full head_dim=64 allocations.
/// Each frequency index controls 2 dims via rotate_half pairing, so the 32 frequency
/// slots are split [12, 10, 10] for temporal (T), height (H), and width (W) using
/// an interleaved step-3 pattern.
///
/// For text-only tokens, all 3 position dimensions are identical,
/// which reduces mRoPE to standard 1D RoPE.
pub struct MRopeEmbedding<B: Backend> {
    /// Inverse frequencies for each dim, shape: [head_dim/2]
    inv_freq: Vec<f64>,
    /// Assignment of each frequency dim to T(0), H(1), or W(2)
    dim_assignment: Vec<usize>,
    /// mrope_section sizes [t_dims, h_dims, w_dims] (e.g., [24, 20, 20])
    pub mrope_section: [usize; 3],
    pub head_dim: usize,
    _device: Device<B>,
}

impl<B: Backend> MRopeEmbedding<B> {
    /// Create mRoPE tables.
    ///
    /// - `mrope_section`: [t_dims, h_dims, w_dims] specifying how many full dims per component
    ///   (e.g., [24, 20, 20] for head_dim=64). Each frequency index controls 2 dims via rotate_half,
    ///   so these are halved internally to get frequency-slot counts [12, 10, 10].
    /// - `rope_theta`: base frequency (5,000,000 for Qwen3-VL)
    /// - `head_dim`: per-head dimension (64 for Qwen3-VL-2B)
    pub fn new(
        mrope_section: [usize; 3],
        rope_theta: f64,
        head_dim: usize,
        device: &Device<B>,
    ) -> Self {
        let half_dim = head_dim / 2;

        // Compute inverse frequencies in f64
        let inv_freq: Vec<f64> = (0..half_dim)
            .map(|i| 1.0 / rope_theta.powf(i as f64 * 2.0 / head_dim as f64))
            .collect();

        // Build interleaved dim assignment
        // mrope_section counts full-dim allocations; each frequency controls 2 dims
        // via rotate_half pairing, so halve to get frequency-slot counts.
        let half_sections = [
            mrope_section[0] / 2,
            mrope_section[1] / 2,
            mrope_section[2] / 2,
        ];
        let dim_assignment = build_dim_assignment(&half_sections, half_dim);

        Self {
            inv_freq,
            dim_assignment,
            mrope_section,
            head_dim,
            _device: device.clone(),
        }
    }

    /// Compute cos/sin for given 3D position IDs.
    ///
    /// - `position_ids`: `[3, seq_len]` — T, H, W positions per token
    ///
    /// Returns `(cos, sin)` each of shape `[seq_len, head_dim]`.
    pub fn compute_cos_sin(
        &self,
        position_ids: &[Vec<usize>],
        device: &Device<B>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        assert_eq!(
            position_ids.len(),
            3,
            "position_ids must have 3 rows (T, H, W)"
        );
        let seq_len = position_ids[0].len();
        let half_dim = self.head_dim / 2;

        let mut cos_data = Vec::with_capacity(seq_len * self.head_dim);
        let mut sin_data = Vec::with_capacity(seq_len * self.head_dim);

        #[allow(clippy::needless_range_loop)]
        for pos_idx in 0..seq_len {
            for freq_idx in 0..half_dim {
                let component = self.dim_assignment[freq_idx]; // 0=T, 1=H, 2=W
                let position = position_ids[component][pos_idx] as f64;
                let angle = position * self.inv_freq[freq_idx];
                cos_data.push(angle.cos() as f32);
                sin_data.push(angle.sin() as f32);
            }
            // Duplicate to full head_dim (standard RoPE convention)
            let start = cos_data.len() - half_dim;
            cos_data.extend_from_within(start..);
            sin_data.extend_from_within(start..);
        }

        let cos =
            Tensor::<B, 1>::from_floats(&cos_data[..], device).reshape([seq_len, self.head_dim]);
        let sin =
            Tensor::<B, 1>::from_floats(&sin_data[..], device).reshape([seq_len, self.head_dim]);

        (cos, sin)
    }
}

/// Build the dimension assignment mapping: freq_idx -> component (0=T, 1=H, 2=W).
///
/// Uses interleaved step-3 pattern matching Qwen3-VL's implementation.
fn build_dim_assignment(mrope_section: &[usize; 3], half_dim: usize) -> Vec<usize> {
    let mut assignment = vec![0usize; half_dim];
    let mut counts = [0usize; 3]; // how many dims assigned to each component

    #[allow(clippy::needless_range_loop)]
    for freq_idx in 0..half_dim {
        // Interleaved pattern: freq_idx % 3 gives candidate component
        let candidate = freq_idx % 3;

        // If this component still has capacity, assign it
        if counts[candidate] < mrope_section[candidate] {
            assignment[freq_idx] = candidate;
            counts[candidate] += 1;
        } else {
            // Find a component that still has capacity
            let mut found = false;
            for c in 0..3 {
                if counts[c] < mrope_section[c] {
                    assignment[freq_idx] = c;
                    counts[c] += 1;
                    found = true;
                    break;
                }
            }
            if !found {
                // All components full — assign to T (shouldn't happen if sections sum to half_dim)
                assignment[freq_idx] = 0;
            }
        }
    }

    assignment
}

/// Build 3D position IDs for a mixed text+vision sequence.
///
/// - `token_ids`: the full token sequence
/// - `image_token_id`: the placeholder token for image patches
/// - `grid_thws`: `[(t, h, w)]` for each image in the sequence
///
/// Returns `[3, seq_len]` position IDs (T, H, W).
///
/// Text tokens use the same position for all 3 dims (standard 1D RoPE).
/// Vision tokens get 3D positions based on their temporal/height/width position in the grid.
pub fn build_position_ids(
    token_ids: &[u32],
    image_token_id: u32,
    grid_thws: &[(usize, usize, usize)],
) -> Vec<Vec<usize>> {
    let seq_len = token_ids.len();
    let mut t_pos = vec![0usize; seq_len];
    let mut h_pos = vec![0usize; seq_len];
    let mut w_pos = vec![0usize; seq_len];

    let mut text_position = 0usize;
    let mut image_idx = 0;
    let mut i = 0;

    while i < seq_len {
        if token_ids[i] == image_token_id && image_idx < grid_thws.len() {
            let (grid_t, grid_h, grid_w) = grid_thws[image_idx];
            let num_vision_tokens = grid_t * grid_h * grid_w;

            // Assign 3D positions to vision tokens
            for vt in 0..grid_t {
                for vh in 0..grid_h {
                    for vw in 0..grid_w {
                        if i < seq_len && token_ids[i] == image_token_id {
                            t_pos[i] = text_position + vt;
                            h_pos[i] = text_position + vh;
                            w_pos[i] = text_position + vw;
                            i += 1;
                        }
                    }
                }
            }

            // Advance text position past the vision region
            let max_dim = grid_t.max(grid_h).max(grid_w);
            text_position += max_dim;
            image_idx += 1;

            let _ = num_vision_tokens;
        } else {
            // Text token: all 3 dims use the same position
            t_pos[i] = text_position;
            h_pos[i] = text_position;
            w_pos[i] = text_position;
            text_position += 1;
            i += 1;
        }
    }

    vec![t_pos, h_pos, w_pos]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn device() -> <B as Backend>::Device {
        Default::default()
    }

    #[test]
    fn dim_assignment_standard_sections() {
        // mrope_section=[24, 20, 20] is full-dim; halved to [12, 10, 10] for 32 freq slots
        let assignment = build_dim_assignment(&[12, 10, 10], 32);
        assert_eq!(assignment.len(), 32);

        // Count per component
        let mut counts = [0usize; 3];
        for &a in &assignment {
            counts[a] += 1;
        }
        // Should exactly match [12, 10, 10]
        assert_eq!(counts, [12, 10, 10]);
    }

    #[test]
    fn text_only_positions_identical() {
        let token_ids = vec![1, 2, 3, 4, 5];
        let image_token_id = 99999; // not present
        let positions = build_position_ids(&token_ids, image_token_id, &[]);

        assert_eq!(positions.len(), 3);
        assert_eq!(positions[0].len(), 5);

        // All 3 dims should be identical for text-only
        for (i, _) in token_ids.iter().enumerate() {
            assert_eq!(positions[0][i], positions[1][i]);
            assert_eq!(positions[1][i], positions[2][i]);
            assert_eq!(positions[0][i], i); // monotonically increasing
        }
    }

    #[test]
    fn vision_tokens_have_3d_positions() {
        // Sequence: [text, text, img, img, img, img, text]
        // where img tokens represent a 1x2x2 grid (T=1, H=2, W=2)
        let image_token_id = 100;
        let token_ids = vec![1, 2, 100, 100, 100, 100, 3];
        let grid_thws = vec![(1, 2, 2)];

        let positions = build_position_ids(&token_ids, image_token_id, &grid_thws);

        // First two text tokens: positions 0, 1
        assert_eq!(positions[0][0], 0);
        assert_eq!(positions[0][1], 1);

        // Vision tokens at indices 2-5: T=1, H=2, W=2
        // T positions should all be text_pos+0 = 2
        assert_eq!(positions[0][2], 2); // vt=0
        assert_eq!(positions[0][3], 2); // vt=0
        assert_eq!(positions[0][4], 2); // vt=0
        assert_eq!(positions[0][5], 2); // vt=0

        // H positions: vh cycles 0,0,1,1 (row-major: vh changes slower)
        assert_eq!(positions[1][2], 2); // vh=0
        assert_eq!(positions[1][3], 2); // vh=0
        assert_eq!(positions[1][4], 3); // vh=1
        assert_eq!(positions[1][5], 3); // vh=1

        // W positions: vw cycles 0,1,0,1
        assert_eq!(positions[2][2], 2); // vw=0
        assert_eq!(positions[2][3], 3); // vw=1
        assert_eq!(positions[2][4], 2); // vw=0
        assert_eq!(positions[2][5], 3); // vw=1

        // Final text token: position after vision region
        // max_dim = max(1,2,2) = 2, so text_position = 2 + 2 = 4
        assert_eq!(positions[0][6], 4);
        assert_eq!(positions[1][6], 4);
        assert_eq!(positions[2][6], 4);
    }

    #[test]
    fn mrope_text_reduces_to_standard_rope() {
        let dev = device();
        let mrope = MRopeEmbedding::<B>::new([24, 20, 20], 5_000_000.0, 64, &dev);

        // Text-only: all 3 dims identical
        let position_ids = vec![vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]];

        let (cos, sin) = mrope.compute_cos_sin(&position_ids, &dev);
        assert_eq!(cos.dims(), [3, 64]);
        assert_eq!(sin.dims(), [3, 64]);

        // cos/sin should be well-defined (not NaN)
        let cos_vals: Vec<f32> = cos.to_data().iter::<f32>().collect();
        for &v in &cos_vals {
            assert!(!v.is_nan(), "cos contains NaN");
        }
    }

    #[test]
    fn mrope_different_positions_differ() {
        let dev = device();
        let mrope = MRopeEmbedding::<B>::new([24, 20, 20], 5_000_000.0, 64, &dev);

        // Two tokens with different spatial positions
        let pos_a = vec![vec![0], vec![0], vec![0]];
        let pos_b = vec![vec![0], vec![1], vec![0]]; // different H

        let (cos_a, _) = mrope.compute_cos_sin(&pos_a, &dev);
        let (cos_b, _) = mrope.compute_cos_sin(&pos_b, &dev);

        let a_vals: Vec<f32> = cos_a.to_data().iter::<f32>().collect();
        let b_vals: Vec<f32> = cos_b.to_data().iter::<f32>().collect();

        let diff: f32 = a_vals
            .iter()
            .zip(b_vals.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "different positions should produce different cos values"
        );
    }
}
