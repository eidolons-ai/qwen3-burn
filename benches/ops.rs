// Criterion benchmarks for qwen3-burn operations.
//
// Uses NdArray (CPU) backend for CI-friendly benchmarks. All dimensions match
// Qwen3-0.6B (hidden=1024, heads=16, kv_heads=8, head_dim=128, intermediate=3072).
//
// GPU sync note: When running with WGPU backend, add `B::sync(&device)` after
// each forward call inside `b.iter()` to ensure GPU work completes before timing
// stops. NdArray is synchronous so no sync is needed here.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use burn::backend::NdArray;
use burn::prelude::*;

use qwen3_burn::bench_internals::*;

type B = NdArray;

fn device() -> <B as Backend>::Device {
    Default::default()
}

// --- Qwen3-0.6B dimensions ---
const HIDDEN: usize = 1024;
const NUM_HEADS: usize = 16;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const INTERMEDIATE: usize = 3072;
const RMS_EPS: f64 = 1e-6;
const ROPE_THETA: f64 = 1_000_000.0;
const MAX_SEQ: usize = 2048;

// --- RmsNorm ---

fn bench_rms_norm(c: &mut Criterion) {
    let dev = device();
    let norm = RmsNorm::<B>::new(HIDDEN, RMS_EPS, &dev);

    let mut group = c.benchmark_group("rms_norm");
    for seq_len in [1, 16, 64, 256, 512] {
        let input = Tensor::<B, 3>::ones([1, seq_len, HIDDEN], &dev);
        group.bench_function(format!("forward/{seq_len}"), |b| {
            b.iter(|| norm.forward(black_box(input.clone())))
        });
    }
    group.finish();
}

// --- RoPE ---

fn bench_rope(c: &mut Criterion) {
    let dev = device();
    let rope = RotaryEmbedding::<B>::new(HEAD_DIM, MAX_SEQ, ROPE_THETA, &dev);

    let mut group = c.benchmark_group("rope");
    for seq_len in [1, 16, 64, 256, 512] {
        let q = Tensor::<B, 4>::ones([1, NUM_HEADS, seq_len, HEAD_DIM], &dev);
        let k = Tensor::<B, 4>::ones([1, NUM_KV_HEADS, seq_len, HEAD_DIM], &dev);
        group.bench_function(format!("apply/{seq_len}"), |b| {
            b.iter(|| rope.apply(black_box(q.clone()), black_box(k.clone()), 0))
        });
    }
    group.finish();
}

// --- FeedForward ---

fn bench_feed_forward(c: &mut Criterion) {
    let dev = device();
    let ff = FeedForward::<B>::new(HIDDEN, INTERMEDIATE, &dev);

    let mut group = c.benchmark_group("feed_forward");
    for seq_len in [1, 16, 64, 256, 512] {
        let input = Tensor::<B, 3>::ones([1, seq_len, HIDDEN], &dev);
        group.bench_function(format!("forward/{seq_len}"), |b| {
            b.iter(|| ff.forward(black_box(input.clone())))
        });
    }
    group.finish();
}

// --- MoE Layer ---
// Uses 8 experts (not 128) for tractable CPU benchmarks.

fn bench_moe_layer(c: &mut Criterion) {
    let dev = device();
    let moe = MoeLayer::<B>::new(HIDDEN, 8, 2, INTERMEDIATE / 4, true, &dev);

    let mut group = c.benchmark_group("moe_layer");
    for seq_len in [1, 16, 64] {
        let input = Tensor::<B, 3>::ones([1, seq_len, HIDDEN], &dev);
        group.bench_function(format!("forward/{seq_len}"), |b| {
            b.iter(|| moe.forward(black_box(input.clone())))
        });
    }
    group.finish();
}

// --- Attention ---

fn bench_attention(c: &mut Criterion) {
    let dev = device();
    let attn =
        MultiHeadAttention::<B>::new(HIDDEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, RMS_EPS, &dev);
    let rope = RotaryEmbedding::<B>::new(HEAD_DIM, MAX_SEQ, ROPE_THETA, &dev);

    let mut group = c.benchmark_group("attention");

    // Prefill benchmarks: process seq_len tokens from scratch
    for seq_len in [1, 16, 64, 256, 512] {
        group.bench_function(format!("prefill/{seq_len}"), |b| {
            b.iter_batched(
                || {
                    let cache =
                        AttentionKvCache::<B>::new(1, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM, &dev);
                    let input = Tensor::<B, 3>::ones([1, seq_len, HIDDEN], &dev);
                    let mask = build_causal_mask::<B>(seq_len, seq_len, &dev);
                    (cache, input, mask)
                },
                |(mut cache, input, mask)| {
                    attn.forward(
                        black_box(input),
                        &rope,
                        Some(black_box(mask)),
                        &mut cache,
                        0,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    }

    // Decode benchmarks: pre-fill cache to `past` tokens, then decode 1 token
    for past in [16, 64, 256, 512] {
        group.bench_function(format!("decode/{past}"), |b| {
            b.iter_batched(
                || {
                    let mut cache =
                        AttentionKvCache::<B>::new(1, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM, &dev);
                    // Pre-fill cache
                    let prefill_input = Tensor::<B, 3>::ones([1, past, HIDDEN], &dev);
                    let prefill_mask = build_causal_mask::<B>(past, past, &dev);
                    attn.forward(prefill_input, &rope, Some(prefill_mask), &mut cache, 0);
                    // Prepare decode input
                    let decode_input = Tensor::<B, 3>::ones([1, 1, HIDDEN], &dev);
                    (cache, decode_input)
                },
                |(mut cache, input)| attn.forward(black_box(input), &rope, None, &mut cache, past),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

// --- TransformerBlock ---

fn bench_transformer_block(c: &mut Criterion) {
    let dev = device();
    let block = TransformerBlock::<B>::new(
        HIDDEN,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        INTERMEDIATE,
        RMS_EPS,
        &dev,
    );
    let rope = RotaryEmbedding::<B>::new(HEAD_DIM, MAX_SEQ, ROPE_THETA, &dev);

    let mut group = c.benchmark_group("transformer_block");

    // Prefill
    for seq_len in [1, 16, 64, 256, 512] {
        group.bench_function(format!("prefill/{seq_len}"), |b| {
            b.iter_batched(
                || {
                    let cache =
                        AttentionKvCache::<B>::new(1, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM, &dev);
                    let input = Tensor::<B, 3>::ones([1, seq_len, HIDDEN], &dev);
                    let mask = build_causal_mask::<B>(seq_len, seq_len, &dev);
                    (cache, input, mask)
                },
                |(mut cache, input, mask)| {
                    block.forward(
                        black_box(input),
                        &rope,
                        Some(black_box(mask)),
                        &mut cache,
                        0,
                    )
                },
                BatchSize::SmallInput,
            )
        });
    }

    // Decode
    for past in [16, 64, 256, 512] {
        group.bench_function(format!("decode/{past}"), |b| {
            b.iter_batched(
                || {
                    let mut cache =
                        AttentionKvCache::<B>::new(1, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM, &dev);
                    let prefill_input = Tensor::<B, 3>::ones([1, past, HIDDEN], &dev);
                    let prefill_mask = build_causal_mask::<B>(past, past, &dev);
                    block.forward(prefill_input, &rope, Some(prefill_mask), &mut cache, 0);
                    let decode_input = Tensor::<B, 3>::ones([1, 1, HIDDEN], &dev);
                    (cache, decode_input)
                },
                |(mut cache, input)| block.forward(black_box(input), &rope, None, &mut cache, past),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

// --- Causal Mask ---

fn bench_causal_mask(c: &mut Criterion) {
    let dev = device();

    let mut group = c.benchmark_group("causal_mask");
    for seq_len in [64, 256, 512, 1024] {
        group.bench_function(format!("build/{seq_len}"), |b| {
            b.iter(|| build_causal_mask::<B>(black_box(seq_len), black_box(seq_len), &dev))
        });
    }
    group.finish();
}

// --- KV Cache ---

fn bench_kv_cache(c: &mut Criterion) {
    let dev = device();

    let mut group = c.benchmark_group("kv_cache");

    // Append benchmarks: append seq_len tokens to a fresh cache
    for seq_len in [1, 16, 64, 256, 512] {
        group.bench_function(format!("append/{seq_len}"), |b| {
            b.iter_batched(
                || {
                    let cache = KvCache::<B>::new(1, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM, &dev);
                    let tensor = Tensor::<B, 4>::ones([1, NUM_KV_HEADS, seq_len, HEAD_DIM], &dev);
                    (cache, tensor)
                },
                |(mut cache, tensor): (KvCache<B>, _)| cache.forward(black_box(tensor)),
                BatchSize::SmallInput,
            )
        });
    }

    // Decode append: pre-fill cache to `depth` tokens, then append 1 token
    for depth in [16, 64, 256, 512] {
        group.bench_function(format!("decode_append/{depth}"), |b| {
            b.iter_batched(
                || {
                    let mut cache = KvCache::<B>::new(1, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM, &dev);
                    let prefill = Tensor::<B, 4>::ones([1, NUM_KV_HEADS, depth, HEAD_DIM], &dev);
                    cache.forward(prefill);
                    let token = Tensor::<B, 4>::ones([1, NUM_KV_HEADS, 1, HEAD_DIM], &dev);
                    (cache, token)
                },
                |(mut cache, token): (KvCache<B>, _)| cache.forward(black_box(token)),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn config() -> Criterion {
    Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(3))
        .measurement_time(std::time::Duration::from_secs(5))
        .sample_size(50)
}

criterion_group! {
    name = benches;
    config = config();
    targets =
        bench_rms_norm,
        bench_rope,
        bench_feed_forward,
        bench_moe_layer,
        bench_attention,
        bench_transformer_block,
        bench_causal_mask,
        bench_kv_cache,
}
criterion_main!(benches);
