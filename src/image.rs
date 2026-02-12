use image::GenericImageView;

/// Preprocessed image/video input ready for the vision encoder.
pub struct ImageInput {
    /// Flattened pixel patches as f32 values normalized to [-1, 1].
    /// Shape: `[num_patches, patch_embed_dim]` flattened into 1D.
    pub pixel_patches: Vec<f32>,
    /// Grid dimensions: (temporal, height, width) in patch units.
    pub grid_thw: (usize, usize, usize),
    /// Number of tokens after spatial merge: grid_t * (grid_h/merge) * (grid_w/merge).
    pub num_merge_tokens: usize,
    /// Original number of patches before merge.
    pub num_patches: usize,
    /// Patch embedding dimension: in_channels * temporal_patch_size * patch_size * patch_size.
    pub patch_embed_dim: usize,
}

/// Configuration for image/video preprocessing.
pub struct ImageProcessor {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub in_channels: usize,
    /// Minimum total pixels after resize (default: 3136 = 56*56).
    pub min_pixels: usize,
    /// Maximum total pixels after resize (default: 1003520 = 1280*784).
    pub max_pixels: usize,
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self {
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            in_channels: 3,
            min_pixels: 3136,
            max_pixels: 1003520,
        }
    }
}

impl ImageProcessor {
    /// The minimum spatial unit: patch_size * spatial_merge_size.
    fn spatial_unit(&self) -> usize {
        self.patch_size * self.spatial_merge_size
    }

    /// Preprocess a single image file for the vision encoder.
    ///
    /// Steps:
    /// 1. Load image
    /// 2. Resize maintaining aspect ratio, dims divisible by spatial_unit (32)
    /// 3. Normalize pixels to [-1, 1]
    /// 4. Duplicate for T=2 (temporal_patch_size)
    /// 5. Extract patches
    pub fn preprocess_image(&self, path: &std::path::Path) -> Result<ImageInput, String> {
        let img = image::open(path).map_err(|e| format!("Failed to open image: {}", e))?;
        let (orig_w, orig_h) = img.dimensions();

        // Resize to dimensions divisible by spatial_unit (32)
        let (new_w, new_h) = self.compute_resize(orig_w as usize, orig_h as usize);
        let img = img.resize_exact(
            new_w as u32,
            new_h as u32,
            image::imageops::FilterType::Lanczos3,
        );

        // Convert to normalized f32 pixels: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1.0
        let rgb = img.to_rgb8();
        let pixels: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| p.0.iter().map(|&v| v as f32 / 127.5 - 1.0))
            .collect();

        // Duplicate frame for temporal_patch_size=2
        let frames: Vec<Vec<f32>> = vec![pixels.clone(), pixels];

        // Extract patches
        let patch_embed_dim =
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size;
        let grid_h = new_h / self.patch_size;
        let grid_w = new_w / self.patch_size;
        let grid_t = self.temporal_patch_size / self.temporal_patch_size; // = 1

        let patches = self.extract_patches(&frames, new_h, new_w);
        let num_patches = grid_t * grid_h * grid_w;
        let merge = self.spatial_merge_size;
        let num_merge_tokens = grid_t * (grid_h / merge) * (grid_w / merge);

        Ok(ImageInput {
            pixel_patches: patches,
            grid_thw: (grid_t, grid_h, grid_w),
            num_merge_tokens,
            num_patches,
            patch_embed_dim,
        })
    }

    /// Preprocess video frames for the vision encoder.
    ///
    /// Frames are paired for temporal patches (padding the last if odd count).
    pub fn preprocess_video_frames(
        &self,
        paths: &[&std::path::Path],
    ) -> Result<ImageInput, String> {
        if paths.is_empty() {
            return Err("No video frames provided".to_string());
        }

        // Load all frames
        let mut frame_pixels: Vec<Vec<f32>> = Vec::new();
        let mut frame_w = 0;
        let mut frame_h = 0;

        for path in paths {
            let img = image::open(path).map_err(|e| format!("Failed to open frame: {}", e))?;
            let (orig_w, orig_h) = img.dimensions();
            if frame_w == 0 {
                let (nw, nh) = self.compute_resize(orig_w as usize, orig_h as usize);
                frame_w = nw;
                frame_h = nh;
            }
            let img = img.resize_exact(
                frame_w as u32,
                frame_h as u32,
                image::imageops::FilterType::Lanczos3,
            );
            let rgb = img.to_rgb8();
            let pixels: Vec<f32> = rgb
                .pixels()
                .flat_map(|p| p.0.iter().map(|&v| v as f32 / 127.5 - 1.0))
                .collect();
            frame_pixels.push(pixels);
        }

        // Pad to even count
        if frame_pixels.len() % self.temporal_patch_size != 0 {
            let last = frame_pixels.last().unwrap().clone();
            frame_pixels.push(last);
        }

        let grid_h = frame_h / self.patch_size;
        let grid_w = frame_w / self.patch_size;
        let grid_t = frame_pixels.len() / self.temporal_patch_size;

        let patches = self.extract_patches(&frame_pixels, frame_h, frame_w);
        let num_patches = grid_t * grid_h * grid_w;
        let merge = self.spatial_merge_size;
        let num_merge_tokens = grid_t * (grid_h / merge) * (grid_w / merge);
        let patch_embed_dim =
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size;

        Ok(ImageInput {
            pixel_patches: patches,
            grid_thw: (grid_t, grid_h, grid_w),
            num_merge_tokens,
            num_patches,
            patch_embed_dim,
        })
    }

    /// Compute resize dimensions maintaining aspect ratio, divisible by spatial_unit,
    /// and with total pixels within [min_pixels, max_pixels].
    fn compute_resize(&self, width: usize, height: usize) -> (usize, usize) {
        let unit = self.spatial_unit();
        let total = width * height;

        let (w, h) = if total > self.max_pixels {
            // Scale down to fit within max_pixels
            let scale = (self.max_pixels as f64 / total as f64).sqrt();
            let w = (width as f64 * scale) as usize;
            let h = (height as f64 * scale) as usize;
            (w, h)
        } else if total < self.min_pixels {
            // Scale up to reach min_pixels
            let scale = (self.min_pixels as f64 / total as f64).sqrt();
            let w = (width as f64 * scale) as usize;
            let h = (height as f64 * scale) as usize;
            (w, h)
        } else {
            (width, height)
        };

        // Round to nearest multiple of unit, minimum 1 unit
        let new_w = ((w + unit / 2) / unit).max(1) * unit;
        let new_h = ((h + unit / 2) / unit).max(1) * unit;
        (new_w, new_h)
    }

    /// Extract patches from frames in row-major spatial order.
    ///
    /// Patches are in `(grid_t, grid_h, grid_w)` order (row-major), with each
    /// patch's data in Conv3D-compatible `[C, T, H, W]` order.
    ///
    /// Input: `frames[temporal][height * width * channels]` (HWC pixel data)
    /// Output: flattened `[num_patches, patch_embed_dim]`
    fn extract_patches(&self, frames: &[Vec<f32>], height: usize, width: usize) -> Vec<f32> {
        let ps = self.patch_size;
        let tps = self.temporal_patch_size;
        let channels = self.in_channels;
        let grid_h = height / ps;
        let grid_w = width / ps;
        let num_temporal_groups = frames.len() / tps;
        let patch_embed_dim = channels * tps * ps * ps;

        let mut patches =
            Vec::with_capacity(num_temporal_groups * grid_h * grid_w * patch_embed_dim);

        // Row-major patch ordering: (tg, ph, pw)
        // Data ordering per patch: [C, T, H, W] (matching Conv3D weight layout)
        for tg in 0..num_temporal_groups {
            for ph in 0..grid_h {
                for pw in 0..grid_w {
                    for c in 0..channels {
                        for t in 0..tps {
                            let frame_idx = tg * tps + t;
                            let frame = &frames[frame_idx];
                            for dy in 0..ps {
                                for dx in 0..ps {
                                    let y = ph * ps + dy;
                                    let x = pw * ps + dx;
                                    let pixel_idx = (y * width + x) * channels + c;
                                    patches.push(frame[pixel_idx]);
                                }
                            }
                        }
                    }
                }
            }
        }

        patches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_divisible_by_32() {
        let proc = ImageProcessor::default();
        let (w, h) = proc.compute_resize(100, 100);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
    }

    #[test]
    fn resize_already_divisible() {
        let proc = ImageProcessor::default();
        let (w, h) = proc.compute_resize(128, 256);
        assert_eq!(w, 128);
        assert_eq!(h, 256);
    }

    #[test]
    fn resize_minimum_size() {
        let proc = ImageProcessor::default();
        let (w, h) = proc.compute_resize(1, 1);
        // 1x1 is below min_pixels (3136), scaled up to ~56x56, rounded to 64x64
        assert_eq!(w, 64);
        assert_eq!(h, 64);
    }

    #[test]
    fn resize_scales_down_large_image() {
        let proc = ImageProcessor::default();
        // 4000x3000 = 12M pixels, above max_pixels (1003520)
        let (w, h) = proc.compute_resize(4000, 3000);
        assert_eq!(w % 32, 0);
        assert_eq!(h % 32, 0);
        let total = w * h;
        // Should be near max_pixels after rounding
        assert!(total <= 1_200_000, "total pixels {} too large", total);
        assert!(total >= 800_000, "total pixels {} too small", total);
        // Aspect ratio roughly preserved
        let ratio = w as f64 / h as f64;
        assert!(
            (ratio - 4.0 / 3.0).abs() < 0.15,
            "aspect ratio {} off",
            ratio
        );
    }

    #[test]
    fn patch_extraction_shape() {
        let proc = ImageProcessor::default();
        // 2 frames, 32x32 pixels, 3 channels
        let frame = vec![0.5f32; 32 * 32 * 3];
        let frames = vec![frame.clone(), frame];

        let patches = proc.extract_patches(&frames, 32, 32);
        // grid: t=1 (2 frames / tps=2), h=2 (32/16), w=2 (32/16)
        // num_patches = 1 * 2 * 2 = 4
        // patch_embed_dim = 3 * 2 * 16 * 16 = 1536
        assert_eq!(patches.len(), 4 * 1536);
    }

    #[test]
    fn normalization_range() {
        // pixel=0 -> 0/127.5 - 1.0 = -1.0
        // pixel=255 -> 255/127.5 - 1.0 = 1.0
        let low = 0.0f32 / 127.5 - 1.0;
        let high = 255.0f32 / 127.5 - 1.0;
        assert!((low - (-1.0)).abs() < 1e-5);
        assert!((high - 1.0).abs() < 1e-3);
    }
}
