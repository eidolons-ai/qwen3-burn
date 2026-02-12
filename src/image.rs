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
    /// Target sampling rate for video in frames per second (default: 2.0).
    pub video_fps: f32,
    /// Minimum number of frames to extract from video (default: 4).
    pub video_min_frames: usize,
    /// Maximum number of frames to extract from video (default: 128).
    pub video_max_frames: usize,
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
            video_fps: 2.0,
            video_min_frames: 4,
            video_max_frames: 128,
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
        let grid_t = 1; // single image = 1 temporal group

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
        if !frame_pixels.len().is_multiple_of(self.temporal_patch_size) {
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

    /// Compute the number of frames to extract from a video of the given duration.
    ///
    /// Frames = `(duration * video_fps).round()`, clamped to `[video_min_frames, video_max_frames]`,
    /// then rounded up to the nearest multiple of `temporal_patch_size` (2).
    pub fn compute_video_frame_count(&self, duration_secs: f64) -> usize {
        let raw = (duration_secs * self.video_fps as f64).round() as usize;
        let clamped = raw.clamp(self.video_min_frames, self.video_max_frames);
        // Round up to nearest multiple of temporal_patch_size
        let tps = self.temporal_patch_size;
        clamped.div_ceil(tps) * tps
    }

    /// Preprocess a video file for the vision encoder.
    ///
    /// Uses `ffprobe` to get the video duration, then `ffmpeg` to extract frames
    /// at a computed FPS. Requires `ffmpeg` and `ffprobe` on PATH.
    ///
    /// Returns an `ImageInput` with the preprocessed video frames.
    pub fn preprocess_video(&self, path: &std::path::Path) -> Result<ImageInput, String> {
        use std::process::Command;

        // 1. Get video duration via ffprobe
        let ffprobe_output = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
            ])
            .arg(path)
            .output()
            .map_err(|e| format!("Failed to run ffprobe (is it installed?): {}", e))?;

        if !ffprobe_output.status.success() {
            let stderr = String::from_utf8_lossy(&ffprobe_output.stderr);
            return Err(format!("ffprobe failed: {}", stderr));
        }

        let duration_str = String::from_utf8_lossy(&ffprobe_output.stdout)
            .trim()
            .to_string();
        let duration_secs: f64 = duration_str
            .parse()
            .map_err(|e| format!("Failed to parse duration '{}': {}", duration_str, e))?;
        eprintln!("Video duration: {:.2}s", duration_secs);

        // 2. Compute frame count
        let num_frames = self.compute_video_frame_count(duration_secs);
        eprintln!("Extracting {} frames...", num_frames);

        // 3. Create temp directory for extracted frames
        let tmp_dir = std::env::temp_dir().join(format!("qwen3vl_video_{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir)
            .map_err(|e| format!("Failed to create temp dir: {}", e))?;

        // 4. Extract frames via ffmpeg
        let effective_fps = num_frames as f64 / duration_secs;
        let output_pattern = tmp_dir.join("frame_%05d.png");
        let ffmpeg_status = Command::new("ffmpeg")
            .args(["-i"])
            .arg(path)
            .args([
                "-vf",
                &format!("fps={}", effective_fps),
                "-frames:v",
                &num_frames.to_string(),
            ])
            .arg(&output_pattern)
            .args(["-loglevel", "error", "-y"])
            .status()
            .map_err(|e| format!("Failed to run ffmpeg (is it installed?): {}", e))?;

        if !ffmpeg_status.success() {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return Err("ffmpeg frame extraction failed".to_string());
        }

        // 5. Collect extracted frame paths (sorted)
        let mut frame_paths: Vec<std::path::PathBuf> = std::fs::read_dir(&tmp_dir)
            .map_err(|e| format!("Failed to read temp dir: {}", e))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|ext| ext == "png"))
            .collect();
        frame_paths.sort();

        if frame_paths.is_empty() {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return Err("ffmpeg extracted 0 frames".to_string());
        }

        eprintln!("Extracted {} frames", frame_paths.len());

        // 6. Process via existing preprocess_video_frames
        let path_refs: Vec<&std::path::Path> = frame_paths.iter().map(|p| p.as_path()).collect();
        let result = self.preprocess_video_frames(&path_refs);

        // 7. Clean up temp directory
        let _ = std::fs::remove_dir_all(&tmp_dir);

        result
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
    fn video_frame_count_basic() {
        let proc = ImageProcessor::default();
        // 10s at 2fps = 20 frames, already even
        assert_eq!(proc.compute_video_frame_count(10.0), 20);
    }

    #[test]
    fn video_frame_count_rounds_to_temporal_patch() {
        let proc = ImageProcessor::default();
        // 2.5s at 2fps = 5 frames, rounded to 5, then ceil to 6 (next multiple of 2)
        assert_eq!(proc.compute_video_frame_count(2.5), 6);
    }

    #[test]
    fn video_frame_count_clamps_min() {
        let proc = ImageProcessor::default();
        // 0.5s at 2fps = 1 frame, clamped to min 4
        assert_eq!(proc.compute_video_frame_count(0.5), 4);
    }

    #[test]
    fn video_frame_count_clamps_max() {
        let proc = ImageProcessor::default();
        // 1000s at 2fps = 2000 frames, clamped to max 128
        assert_eq!(proc.compute_video_frame_count(1000.0), 128);
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
