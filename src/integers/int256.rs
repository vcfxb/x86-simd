//! AVX family 256 bit SIMD values over integer data.

// Allow path statements so that the compiler runs into a reference to the Simd256Integer::_ASSERT_LANES_MATCH_SIZE
// it will fail to compile appropriately/successfully without warning us about "effectless" code.
#![allow(path_statements)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__m256i;

#[cfg(target_arch = "x86")]
use core::arch::x86::__m256i;

#[cfg(any(feature = "std", target_feature = "avx2"))]
use core::any::TypeId;

use core::fmt::Debug;
use core::marker::PhantomData;
use core::mem::{transmute, transmute_copy};
use core::ops::{Add, AddAssign};

use crate::sealed::Sealed;

/// Marker trait implemented on all scalar (primitive) types that can be packed into a [`Simd256Integer`].
pub trait Simd256Scalar:
    Sealed + Sized + Copy + Add<Self, Output = Self> + AddAssign + Default + 'static
{
}

macro_rules! impl_scalars {
    ( $($t:ty $(| $extra:ident )*)* ) => {$(
            impl Simd256Scalar for $t {}

            $(
                impl $extra for $t {}
            )*
    )*};
}

impl_scalars! {
    u8
    | Simd256SaturatingAdd

    i8
    | Simd256SaturatingAdd

    u16
    | Simd256SaturatingAdd

    i16
    | Simd256SaturatingAdd

    u32
    i32
    u64
    i64
}

/// Marker trait implemented on all scalar (primitive) types that can be
pub trait Simd256SaturatingAdd: Simd256Scalar {}

/// This type packs integer data into a 256 bit value and attempts to use AVX family instructions if available for all
/// operations.
///
/// If AVX is not determined to be available, this struct has a fallback implementation that will be slower, but still
/// mathematically correct, and may attempt to use SSE family instructions if possible.
#[derive(Clone, Copy, Debug)]
pub struct Simd256Integer<S: Simd256Scalar, const LANES: usize> {
    /// phantom data to make generics are used.
    phantom: PhantomData<[S; LANES]>,

    /// Underlying bit storage.
    pub inner: Simd256IntegerInner,
}

/// The internal representation for 256-bit integer data SIMD values used by [Simd256Integer].
#[derive(Clone, Copy)]
pub union Simd256IntegerInner {
    /// If AVX is available, this field of the union will be active and contain am [__m256i] value.
    #[cfg(any(feature = "std", target_feature = "avx"))]
    pub avx: __m256i,

    /// Fallback representation if we cannot confirm that AVX instructions are available.
    /// This will be slower than the AVX version, but at least still mathematically correct.
    pub fallback: [u8; size_of::<__m256i>() / size_of::<u8>()],
}

impl Debug for Simd256IntegerInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        #[cfg(feature = "std")]
        if std::is_x86_feature_detected!("avx") {
            return f
                .debug_struct(stringify!(Simd256IntegerInner))
                // SAFETY: We just checked that AVX is supported -- if it is, we are using it.
                .field("avx", &unsafe { self.avx })
                .finish();
        }

        // If std is disabled, try to use compiler flags to determine AVX support.
        #[cfg(all(not(feature = "std"), target_feature = "avx"))]
        {
            return f
                .debug_struct(stringify!(Simd256IntegerInner))
                // SAFETY: We just checked that AVX is supported -- if it is, we are using it.
                .field("avx", &unsafe { self.avx })
                .finish();
        }

        // If we haven't returned yet, we're using the fallback representation.
        f.debug_struct(stringify!(Simd256IntegerInner))
            // SAFETY: We checked above that AVX is not available.
            .field("fallback", &unsafe { self.fallback.as_ref() })
            .finish()
    }
}

impl<S: Simd256Scalar, const LANES: usize> Simd256Integer<S, LANES> {
    /// Compile-time assertion that number of lanes size of SIMD vector.
    const _MENTION_ME_TO_ASSERT_LANES_MATCH_SIZE: () = assert!(
        LANES == size_of::<__m256i>() / size_of::<S>(),
        "The number of lanes needs to be consistent with the size of the SIMD vector for the scalar type."
    );

    /// Construct a [Simd256Integer] value from an array of scalar values.
    ///
    /// This function will eventually be made `const` after <https://github.com/rust-lang/rust/issues/80384> is
    /// resolved (it can't currently since the compiler can't/doesn't prove that S cannot contain an unsafe cell).
    ///
    /// Note that this function will fail at compile time if you attempt to construct a [Simd256Integer] with
    /// a number of `LANES` inconsistent with the size of the scalar type `S`. See below:
    /// ```compile_fail
    /// use x86_simd::integers::int256::Simd256Integer;
    /// let splat = Simd256Integer::from_array([0; 50]);
    /// ```
    pub fn from_array(array: [S; LANES]) -> Self {
        // Check that the number of lanes is good (this is a compile-time check triggered by seeing this const).
        Self::_MENTION_ME_TO_ASSERT_LANES_MATCH_SIZE;

        // SAFETY: This is not amazing, but these types are the same size (as asserted at compile time above), and
        // are both just plain old data, so this transmute should do what we like, and we can confirm with unit tests.
        unsafe { transmute_copy(&array) }
    }

    /// "splat" a given scalar across all lanes of this SIMD value.
    pub fn splat(s: S) -> Self {
        Simd256Integer::from_array([s; LANES])
    }

    /// Take `LANES` items from an [Iterator] into the lanes of a [Simd256Integer].
    ///
    /// If the [Iterator::next] ever returns [`None`], immediately stop and return [`None`] (this means that
    /// the iterator will be partially consumed if the end of it is reached before a SIMD vector can be filled).
    pub fn try_from_iter(iter: &mut impl Iterator<Item = S>) -> Option<Self> {
        // Check that the number of lanes is good (this is a compile-time check triggered by seeing this const).
        Self::_MENTION_ME_TO_ASSERT_LANES_MATCH_SIZE;

        // We can use default here since it's always overwritten (so it doesn't matter what it is) and it also should
        // be zero for all the scalar types this is Sealed to.
        let mut array: [S; LANES] = [Default::default(); LANES];

        #[allow(clippy::needless_range_loop)]
        for i in 0..LANES {
            array[i] = iter.next()?;
        }

        Some(Simd256Integer::from_array(array))
    }

    /// Turn this [Simd256Integer] into an array of its lanes.
    pub const fn to_array(self) -> [S; LANES] {
        // Check that the number of lanes is good (this is a compile-time check triggered by seeing this const).
        Self::_MENTION_ME_TO_ASSERT_LANES_MATCH_SIZE;

        // SAFETY: This is safe/acceptable for the same reasons as from_array.
        unsafe { transmute_copy(&self) }
    }

    /// Get a reference to the underlying data of this SIMD value as an array.
    pub fn as_array_ref(&self) -> &[S; LANES] {
        // Check that the number of lanes is good (this is a compile-time check triggered by seeing this const).
        Self::_MENTION_ME_TO_ASSERT_LANES_MATCH_SIZE;

        // SAFETY: This is valid because the reciever type has a constant/known size, and therefore should not be
        // a wide pointer, and because the lifetimes/liveness guarantees against this struct's underlying
        // representation continue to hold true for the returned. All this in addtion to the above state reasons under
        // (to|from)_array.
        unsafe { transmute(self) }
    }

    /// Wrap a given intrinsic value with this type.
    #[inline(always)]
    pub const fn from_intrinsic(intrinsic: __m256i) -> Self {
        // Check that the number of lanes is good (this is a compile-time check triggered by seeing this const).
        Self::_MENTION_ME_TO_ASSERT_LANES_MATCH_SIZE;

        // SAFETY: Honestly you should be used to me transmuting between plain data of the same sizes at this point.
        unsafe { transmute(intrinsic) }
    }

    /// "vertically" Add two SIMD values to eachother using AVX2 instructions.
    ///
    /// "vertical" means each lane of the resulting SIMD value contains the sum of the coresponding
    /// lanes of `a` and `b`.
    ///
    /// # Safety
    /// The caller must ensure that AVX2 CPU features are supported, otherwise calling this function will
    /// execute unsupoorted instructions (which is immediate undefined behaviour).
    #[cfg(any(feature = "std", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn avx2_vertical_add(a: Self, b: Self) -> Self {
        // Check that the number of lanes is good (this is a compile-time check triggered by seeing this const).
        Self::_MENTION_ME_TO_ASSERT_LANES_MATCH_SIZE;

        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        let result = match size_of::<S>() {
            1 => _mm256_add_epi8(a.inner.avx, b.inner.avx),
            2 => _mm256_add_epi16(a.inner.avx, b.inner.avx),
            4 => _mm256_add_epi32(a.inner.avx, b.inner.avx),
            8 => _mm256_add_epi64(a.inner.avx, b.inner.avx),
            _ => crate::unreachable_uncheched_on_release(),
        };

        Self::from_intrinsic(result)
    }

    /// Saturating vertical SIMD add using AVX2 instructions.
    /// Saturated adds are generally slower than [Self::avx2_vertical_add] so use only when needed if you care about
    /// performance (if you don't care about performance then why are you using this SIMD library anyway).
    ///
    /// # Safety
    /// The caller must ensure that AVX2 CPU features are supported, otherwise calling this function will
    /// execute unsupoorted instructions (which is immediate undefined behaviour).
    #[cfg(any(feature = "std", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn avx2_vertical_saturating_add(a: Self, b: Self) -> Self
    where
        S: Simd256SaturatingAdd,
    {
        // Check that the number of lanes is good (this is a compile-time check triggered by seeing this const).
        Self::_MENTION_ME_TO_ASSERT_LANES_MATCH_SIZE;

        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        // I do not love using `TypeId` here but the compiler will optimize it out according to `cargo asm`
        // https://crates.io/crates/cargo-show-asm, so this is how it works for now I suppose.
        let result = match size_of::<S>() {
            1 if TypeId::of::<S>() == TypeId::of::<u8>() => {
                _mm256_adds_epu8(a.inner.avx, b.inner.avx)
            }
            1 if TypeId::of::<S>() == TypeId::of::<i8>() => {
                _mm256_adds_epi8(a.inner.avx, b.inner.avx)
            }
            2 if TypeId::of::<S>() == TypeId::of::<u16>() => {
                _mm256_adds_epu16(a.inner.avx, b.inner.avx)
            }
            2 if TypeId::of::<S>() == TypeId::of::<i16>() => {
                _mm256_adds_epi16(a.inner.avx, b.inner.avx)
            }
            _ => crate::unreachable_uncheched_on_release(),
        };

        Self::from_intrinsic(result)
    }
}

impl<S: Simd256Scalar, const LANES: usize> core::ops::Add for Simd256Integer<S, LANES> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // First attempt to use avx2 based SIMD add.
        #[cfg(feature = "std")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: We just checked if the CPU supports AVX2.
            return unsafe { Simd256Integer::avx2_vertical_add(self, rhs) };
        }

        #[cfg(target_feature = "avx2")]
        // SAFETY: We statically check if the CPU supports AVX2.
        return unsafe { Simd256Integer::avx2_vertical_add(self, rhs) };

        // If neither of the above has returned already, use a fallback.
        // This is fully safe, thanks to guarantees made elsewhere, and is just an iterative vertical add across two
        // scalar arrays.
        let mut result: [S; LANES] = [Default::default(); LANES];
        let a = self.to_array();
        let b = rhs.to_array();

        for i in 0..result.len() {
            result[i] = a[i] + b[i];
        }

        Simd256Integer::from_array(result)
    }
}

// #[target_feature(enable = "avx2")]
// pub unsafe fn avx_sadd_i16(a: Simd256Integer<i16, 16>, b: Simd256Integer<i16, 16>) -> Simd256Integer<i16, 16> {
//     Simd256Integer::<i16, 16>::avx2_vertical_saturating_add(a, b)
// }

#[cfg(test)]
mod tests {
    use super::Simd256Integer;

    // This test should fail to compile if you un-comment it.
    // #[test]
    // fn comp_fail() {
    //     let splat = Simd256Integer::from_array([0u8; 100]);
    // }

    #[test]
    fn test_debug() {
        let simd_value = Simd256Integer::<u8, 32>::try_from_iter(&mut (0..32).into_iter()).unwrap();

        println!("{simd_value:#x?}");
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_avx2_add() {
        assert!(is_x86_feature_detected!("avx2"));

        let simd_10x16 = Simd256Integer::<u16, 16>::splat(10);
        let simd_12x16 = Simd256Integer::<u16, 16>::splat(12);

        let added = unsafe { Simd256Integer::avx2_vertical_add(simd_10x16, simd_12x16) };

        assert_eq!(added.to_array(), [22; 16]);
    }
}
