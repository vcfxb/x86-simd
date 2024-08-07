use std::hint::black_box;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use x86_simd::integers::int256::u16x16;
// use core::arch::x86_64::*;

#[inline(always)]
pub fn simd_vertical_add(a: u16x16, b: u16x16) -> u16x16 {    
    // SAFETY: Assume that we have AVX2 -- it should be checked outside this benchmark function.
    unsafe { u16x16::avx2_vertical_add(a, b) }
}

#[inline(always)]
pub fn scalar_vertical_add(a: u16x16, b: u16x16) -> u16x16 {
    let mut result = [0; 16];

    for i in 0..16 {
        result[i] = a.as_array_ref()[i] + b.as_array_ref()[i];
    }

    u16x16::from_array(result)
}

fn bench_sum(c: &mut Criterion) {
    // Make sure we have AVX2.
    assert!(
        is_x86_feature_detected!("avx2"), 
        "AVX2 cpu feature is required to compare performance between AVX2 and scalar math."
    );

    let mut group = c.benchmark_group("Vertical Sum");

    // Use 5 random inputs.
    for _ in 0..5 {
        let mut a = [0; 16];
        let mut b = [0; 16];
        rand::thread_rng().fill(&mut a);
        rand::thread_rng().fill(&mut b);

        let a = u16x16::from_array(a);
        let b = u16x16::from_array(b);

        group.bench_with_input(
            BenchmarkId::new("SIMD (AVX2) Vertical Add", format!("{a:?} + {b:?}")),
            &(a, b),
            |b, i| b.iter(|| {
                black_box(simd_vertical_add(i.0, i.1));
            })
        );
        
        group.bench_with_input(
            BenchmarkId::new("Scalar Vertical Add", format!("{a:?} + {b:?}")),
            &(a, b),
            |b, i| b.iter(|| {
                black_box(scalar_vertical_add(i.0, i.1));
            })
        );
    }

    group.finish();
}

criterion_group!(benches, bench_sum);
criterion_main!(benches);
