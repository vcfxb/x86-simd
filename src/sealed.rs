//! Marker trait to indicate that a trait is not to be implemented outside of this crate.

/// Marker trait to indicate that a trait is not to be implemented outside of this crate.
#[allow(unused)]
pub(crate) trait Sealed {}

macro_rules! impl_sealed {
    ( $($t:ty)* ) => {
        $(
            impl $crate::sealed::Sealed for $t {}
        )*
    };
}

impl_sealed!( u8 i8 u16 i16 u32 i32 u64 i64 u128 i128 f32 f64 );
