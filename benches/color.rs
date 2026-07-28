//! Full-image colour conversion: `Vec::push` vs preallocated slice writes.
//!
//! `convert_rgb_to_ycbcr` and `deinterleave_ycbcr` (src/color.rs) run over
//! every pixel on the encode path and were written as per-pixel `Vec::push`
//! loops. push pays a capacity check per element and blocks vectorization.
//!
//! Run: `cargo bench --bench color`

use zenbench::prelude::*;
use zenjpeg_dispatch::__bench_color::{convert_rgb_to_ycbcr, deinterleave_ycbcr, rgb_to_ycbcr};

fn img(px: usize) -> Vec<u8> {
    let mut s = 0x9e37_79b9u32;
    (0..px * 3)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 24) as u8
        })
        .collect()
}

/// The previous shape, kept so the comparison is against what actually shipped
/// rather than against a strawman.
fn push_rgb_to_ycbcr(rgb: &[u8]) -> Vec<u8> {
    let mut ycbcr = Vec::with_capacity(rgb.len());
    for chunk in rgb.chunks_exact(3) {
        let (y, cb, cr) = rgb_to_ycbcr(chunk[0], chunk[1], chunk[2]);
        ycbcr.push(y);
        ycbcr.push(cb);
        ycbcr.push(cr);
    }
    ycbcr
}

fn push_deinterleave(ycbcr: &[u8], n: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (mut y, mut cb, mut cr) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for chunk in ycbcr.chunks_exact(3) {
        y.push(chunk[0]);
        cb.push(chunk[1]);
        cr.push(chunk[2]);
    }
    (y, cb, cr)
}

fn bench_color(suite: &mut Suite) {
    for &(label, w, h) in &[("1024x1024", 1024usize, 1024usize), ("4096x4096", 4096, 4096)] {
        let px = w * h;
        let rgb: &'static [u8] = Box::leak(img(px).into_boxed_slice());

        suite.compare(format!("rgb_to_ycbcr/{label}"), |g| {
            g.throughput(Throughput::Bytes((px * 3) as u64));
            g.bench("push_was", move |b| b.iter(move || push_rgb_to_ycbcr(rgb)));
            g.bench("slice_now", move |b| {
                b.iter(move || convert_rgb_to_ycbcr(rgb, w, h))
            });
        });

        suite.compare(format!("deinterleave/{label}"), |g| {
            g.throughput(Throughput::Bytes((px * 3) as u64));
            g.bench("push_was", move |b| b.iter(move || push_deinterleave(rgb, px)));
            g.bench("slice_now", move |b| {
                b.iter(move || deinterleave_ycbcr(rgb, w, h))
            });
        });
    }
}

zenbench::main!(bench_color);
