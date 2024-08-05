//! AVX family 256 bit SIMD values over integer data.

// Allow path statements so that the compiler runs into a reference to the Simd256Integer::_ASSERT_LANES_MATCH_SIZE
// it will fail to compile appropriately/successfully without warning us about "effectless" code.
#![allow(path_statements)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__m256i;

#[cfg(target_arch = "x86")]
use core::arch::x86::__m256i;

use core::any::TypeId;
use core::fmt::Debug;
use core::marker::PhantomData;
use core::mem::{transmute, transmute_copy};
use core::ops::{Add, AddAssign};

use crate::sealed::Sealed;

/// Marker trait implemented on all scalar (primitive) types that can be packed into a [`Simd256Integer`].
pub trait Simd256Scalar:
    Sealed + Sized + Copy + Add<Self, Output = Self> + AddAssign + Default + 'static
{}

/// Marker trait implemented on all scalar (primitive) types that support saturating addition AVX2 operations.
pub trait Simd256SaturatingAdd: Simd256Scalar {}

/// Marker trait on all scalar (primitive) types that support the absolute value AVX2 operations.
pub trait Simd256IntegerAbs: Simd256Scalar {}

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
    | Simd256IntegerAbs

    u16
    | Simd256SaturatingAdd

    i16
    | Simd256SaturatingAdd
    | Simd256IntegerAbs 

    u32

    i32
    | Simd256IntegerAbs
    
    u64
    i64
}

/// 32 [u8] values in a SIMD vector backed by AVX-family operations or a fallback.
#[allow(non_camel_case_types)]
pub type u8x32 = Simd256Integer<u8, 32>;

/// 32 [i8] values in a SIMD vector backed by AVX-family operations or a fallback.
#[allow(non_camel_case_types)]
pub type i8x32 = Simd256Integer<i8, 32>;

/// 16 [u16] values in a SIMD vector backed by AVX-family operations or a fallback.
#[allow(non_camel_case_types)]
pub type u16x16 = Simd256Integer<u16, 16>;

/// 16 [i16] values in a SIMD vector backed by AVX-family operations or a fallback.
#[allow(non_camel_case_types)]
pub type i16x16 = Simd256Integer<i16, 16>;

/// 8 [u32] values in a SIMD vector backed by AVX-family operations or a fallback.
#[allow(non_camel_case_types)]
pub type u32x8 = Simd256Integer<u32, 8>;

/// 8 [i32] values in a SIMD vector backed by AVX-family operations or a fallback.
#[allow(non_camel_case_types)]
pub type i32x8 = Simd256Integer<i32, 8>;

/// 4 [u64] values in a SIMD vector backed by AVX-family operations or a fallback.
#[allow(non_camel_case_types)]
pub type u64x4 = Simd256Integer<u64, 4>;

/// 4 [i64] values in a SIMD vector backed by AVX-family operations or a fallback.
#[allow(non_camel_case_types)]
pub type i64x4 = Simd256Integer<i64, 4>;

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

    /// Check if the element type of this [Simd256Integer] (`S`) matches the given type `T` using [`core::any::TypeId`].
    #[inline(always)]
    pub fn element_type_is<T: Simd256Scalar>() -> bool {
        TypeId::of::<S>() == TypeId::of::<T>()
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
            1 if Self::element_type_is::<u8>() => {
                _mm256_adds_epu8(a.inner.avx, b.inner.avx)
            }
            
            1 if Self::element_type_is::<i8>() => {
                _mm256_adds_epi8(a.inner.avx, b.inner.avx)
            }
            
            2 if Self::element_type_is::<u16>() => {
                _mm256_adds_epu16(a.inner.avx, b.inner.avx)
            }
            
            2 if Self::element_type_is::<i16>() => {
                _mm256_adds_epi16(a.inner.avx, b.inner.avx)
            }

            _ => crate::unreachable_uncheched_on_release(),
        };

        Self::from_intrinsic(result)
    }

    /// Saturating add on two SIMD vectors backed by AVX2 operations or using fallback iterative/scalar instructions.
    pub fn saturating_add(a: Self, b: Self) -> Self
    where S: Simd256SaturatingAdd
    {
        Self::_MENTION_ME_TO_ASSERT_LANES_MATCH_SIZE;

        #[cfg(target_feature = "avx2")]
        // SAFETY: We checked if the CPU supports AVX2.
        return unsafe { Self::avx2_vertical_saturating_add(a, b) };

        #[cfg(feature = "std")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: We checked if the CPU supports AVX2.
            return unsafe { Self::avx2_vertical_saturating_add(a, b) };
        }


        // Hate to use type-of again here but it's safe and gets compiled away on release.
        // This is all fallback for when avx2 is not available.
        match size_of::<S>() {
            1 if Self::element_type_is::<u8>() => {
                // SAFETY: We have just checked the type.
                let a = unsafe { transmute::<_, u8x32>(a) }.to_array();
                let b = unsafe { transmute::<_, u8x32>(b) }.to_array();
                let mut result: [u8; 32] = [0; 32];

                for i in 0..LANES {
                    result[i] = u8::saturating_add(a[i], b[i]);
                }

                unsafe { transmute(result) }
            }

            1 if Self::element_type_is::<i8>() => {
                // SAFETY: We have just checked the type.
                let a = unsafe { transmute::<_, i8x32>(a) }.to_array();
                let b = unsafe { transmute::<_, i8x32>(b) }.to_array();
                let mut result: [i8; 32] = [0; 32];

                for i in 0..LANES {
                    result[i] = i8::saturating_add(a[i], b[i]);
                }

                unsafe { transmute(result) }
            }
            
            2 if Self::element_type_is::<u16>() => {
                // SAFETY: We have just checked the type.
                let a = unsafe { transmute::<_, u16x16>(a) }.to_array();
                let b = unsafe { transmute::<_, u16x16>(b) }.to_array();
                let mut result: [u16; 16] = [0; 16];

                for i in 0..LANES {
                    result[i] = u16::saturating_add(a[i], b[i]);
                }

                unsafe { transmute(result) }
            }

            2 if Self::element_type_is::<i16>() => {
                // SAFETY: We have just checked the type.
                let a = unsafe { transmute::<_, i16x16>(a) }.to_array();
                let b = unsafe { transmute::<_, i16x16>(b) }.to_array();
                let mut result: [i16; 16] = [0; 16];

                for i in 0..LANES {
                    result[i] = i16::saturating_add(a[i], b[i]);
                }

                unsafe { transmute(result) }
            }

            // SAFETY: We checked all types that implement AVX2-backed saturating addition, and that trait is Sealed.
            _ => unsafe { crate::unreachable_uncheched_on_release() }
        }
    }


    /// Compare two SIMD vectors for equality of elements vertically. Lanes of the result are defined as so: 
    /// if the elements of the coresponding lane of each of the input vectors are equal, then the output vector will 
    /// have all `1` bits in that lane. If not equal, then all `0` bits.
    /// 
    /// # Safety
    /// The caller must ensure that AVX2 CPU features are supported, otherwise calling this function will
    /// execute unsupoorted instructions (which is immediate undefined behaviour).
    #[cfg(any(feature = "std", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn avx2_vertical_cmp_eq(a: Self, b: Self) -> Self {
        Self::_MENTION_ME_TO_ASSERT_LANES_MATCH_SIZE;

        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        
        let result = match size_of::<S>() {
            1 => _mm256_cmpeq_epi8(a.inner.avx, b.inner.avx),
            2 => _mm256_cmpeq_epi16(a.inner.avx, b.inner.avx),
            4 => _mm256_cmpeq_epi32(a.inner.avx, b.inner.avx),
            8 => _mm256_cmpeq_epi64(a.inner.avx, b.inner.avx),
            _ => crate::unreachable_uncheched_on_release(),
        };

        Self::from_intrinsic(result)
    }

    /// Get the absolute value of each lane of this SIMD vector using AVX2 absolute value intrinsics.
    /// 
    /// # Safety
    /// The caller must ensure that AVX2 CPU features are supported, otherwise calling this function will
    /// execute unsupoorted instructions (which is immediate undefined behaviour).
    #[cfg(any(feature = "std", target_feature = "avx2"))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn avx2_vertical_abs(self) -> Self 
    where S: Simd256IntegerAbs
    {
        Self::_MENTION_ME_TO_ASSERT_LANES_MATCH_SIZE;

        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        
        let result = match size_of::<S>() {
            1 => _mm256_abs_epi8(self.inner.avx),
            2 => _mm256_abs_epi16(self.inner.avx),
            4 => _mm256_abs_epi32(self.inner.avx),
            _ => crate::unreachable_uncheched_on_release(),
        };

        Self::from_intrinsic(result)
    }

    /// Return a SIMD vector containing the absolute value of all of the elements of this SIMD vector.
    pub fn abs(self) -> Self
    where S: Simd256IntegerAbs {
        Self::_MENTION_ME_TO_ASSERT_LANES_MATCH_SIZE;

        #[cfg(target_feature = "avx2")]
        // SAFETY: We checked if the CPU supports AVX2.
        return unsafe { Self::avx2_vertical_abs(self) };

        #[cfg(feature = "std")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: We checked if the CPU supports AVX2.
            return unsafe { Self::avx2_vertical_abs(self) };
        }

        // Fallback if AVX2 is not supported.
        // SAFETY: We match on the size of `S` and know all the types that implement the sealed trait.
        match size_of::<S>() {
            1 => unsafe { 
                let mut array = transmute::<_, i8x32>(self).to_array();

                for element in &mut array {
                    *element = i8::abs(*element);
                }

                transmute(array)
            }

            2 => unsafe { 
                let mut array = transmute::<_, i16x16>(self).to_array();

                for element in &mut array {
                    *element = i16::abs(*element);
                }

                transmute(array)
            }

            
            4 => unsafe { 
                let mut array = transmute::<_, i32x8>(self).to_array();

                for element in &mut array {
                    *element = i32::abs(*element);
                }

                transmute(array)
            }

            _ => unsafe { crate::unreachable_uncheched_on_release() }
        }
    }


}

impl<S: Simd256Scalar, const LANES: usize> core::ops::Add for Simd256Integer<S, LANES> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::_MENTION_ME_TO_ASSERT_LANES_MATCH_SIZE;
    
        #[cfg(target_feature = "avx2")]
        // SAFETY: We statically check if the CPU supports AVX2.
        return unsafe { Simd256Integer::avx2_vertical_add(self, rhs) };

        // Attempt to use avx2 based SIMD add.
        #[cfg(feature = "std")]
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: We just checked if the CPU supports AVX2.
            return unsafe { Simd256Integer::avx2_vertical_add(self, rhs) };
        }

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

// pub fn avx_sadd_i16(a: Simd256Integer<i16, 16>, b: Simd256Integer<i16, 16>) -> Simd256Integer<i16, 16> {
//     Simd256Integer::<i16, 16>::saturating_add(a, b)
// }

#[cfg(test)]
mod tests {
    use core::u16;

    use super::u8x32;
    use super::u16x16;
    use super::i16x16;

    // This test should fail to compile if you un-comment it.
    // #[test]
    // fn comp_fail() {
    //     let splat = Simd256Integer::from_array([0u8; 100]);
    // }

    #[test]
    #[cfg(feature = "std")]
    fn test_debug() {
        let simd_value = u8x32::try_from_iter(&mut (0..32).into_iter()).unwrap();

        println!("{simd_value:#x?}");
    }

    #[test]
    fn test_add() {
        let simd_10x16 = u16x16::splat(10);
        let simd_12x16 = u16x16::splat(12);

        let added =  simd_10x16 + simd_12x16;

        assert_eq!(added.to_array(), [22; 16]);
    }

    #[test]
    fn test_saturating_add() {
        let simd_max = u16x16::splat(u16::MAX);

        assert_eq!(u16x16::saturating_add(simd_max, simd_max).as_array_ref(), simd_max.as_array_ref());
    }

    #[test]
    fn test_abs() {
        let simd_neg1 = i16x16::splat(-1);

        assert_eq!(simd_neg1.abs().to_array(), [1; 16]);
    }
}
