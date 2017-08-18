//! A Rust implementation of the [posit] number system
//!
//! [posit]: http://web.stanford.edu/class/ee380/Abstracts/170201.html
//!
//! # Examples
//!
//! ```
//! // 8-bit posit with ES = 2
//! use posit::P8E2;
//!
//! assert_eq!(P8E2(2.).unwrap() * P8E2(0.5).unwrap(),
//!            P8E2(1.).unwrap());
//! ```
//!
//! # Notes
//!
//! - Only basic arithmetic has been implemented and the implementation has not
//!   been optimized for performance. As a result, posit arithmetic is about 2x
//!   slower than software emulated ieee754 float arithmetic (`P32E6` vs `f32`)
//!   on a Cortex-M3 processor.
//!
//! - Posits with size of up to 32 bits with any value of "ES" are supported.
//!
//! - As there's no NaN representation in the posit system, operations that
//!   would result in a NaN value on the ieee float system `panic!` in the posit
//!   system.
//!
//! - There's only one representation of infinity in the posit system and has no
//!   sign information thus posit numbers can't be sorted; they don't implement
//!   `Ord`.
//!
//! - However, unlike ieee754 floats, posits do implement the `Eq` trait as
//!   there's no NaN value.
//!
//! # References
//!
//! - http://web.stanford.edu/class/ee380/Abstracts/170201-slides.pdf
//! - http://www.johngustafson.net/presentations/Unums2.0.pdf

#![cfg_attr(feature = "const-fn", feature(const_fn))]
#![no_std]

#[cfg(test)]
#[macro_use]
extern crate approx;
extern crate cast;
#[cfg(test)]
#[macro_use]
extern crate quickcheck;
extern crate typenum;

use core::marker::PhantomData;
use core::{f64, fmt, mem, ops, u16, u32, u64, u8};
use core::cmp::{self, Ordering};

use cast::{Error, i16, i32, u16, u32, u64};
use typenum::{Cmp, Less, U0, U1, U16, U17, U2, U3, U32, U4, U5, U6, U7, U8,
              U9, Unsigned};
#[cfg(not(feature = "const-fn"))]
use typenum::{Greater, U33};
#[cfg(feature = "const-fn")]
use typenum::{U10, U11, U12, U13, U14, U15, U18, U19, U20, U21, U22, U23, U24,
              U25, U26, U27, U28, U29, U30, U31};

/// A posit
///
/// - `BITS` is the integer primitive used to store the posit
/// - `NBITS` is the number of bits that the posit uses
/// - `ES` is the *maximum* number of exponent bits that the posit can use
///
/// For convenience, type aliases are provided. The types follow the naming
/// convention: `PxEy` where `x` is the number of bits of the posit and `y` is
/// the ES value of the posit.
pub struct Posit<BITS, NBITS, ES> {
    bits: BITS,
    _marker: PhantomData<(NBITS, ES)>,
}

impl<BITS, NBITS, ES> Clone for Posit<BITS, NBITS, ES>
where
    BITS: Clone,
{
    fn clone(&self) -> Self {
        Posit {
            bits: self.bits.clone(),
            _marker: PhantomData,
        }
    }
}

impl<BITS, NBITS, ES> Copy for Posit<BITS, NBITS, ES>
where
    BITS: Copy,
{
}

impl<BITS, NBITS, ES> Eq for Posit<BITS, NBITS, ES>
where
    BITS: Eq,
{
}

impl<BITS, NBITS, ES> PartialEq for Posit<BITS, NBITS, ES>
where
    BITS: PartialEq,
{
    fn eq(&self, rhs: &Self) -> bool {
        self.bits == rhs.bits
    }
}

macro_rules! posit {
    ($bits:ident, $dbits:ident, $limit:ident, $bits_size:expr) => {
        impl<NBITS, ES> Posit<$bits, NBITS, ES>
            where
                ES: Unsigned,
                NBITS: Unsigned,
        {
            #[cfg(not(feature = "const-fn"))]
            /// Creates a new posit from the bit pattern `bits`
            pub fn new(bits: $bits) -> Self
            where
                ES: Cmp<$limit, Output = Less>,
                NBITS: Cmp<$limit, Output = Less> + Cmp<U2, Output = Greater>,
            {
                Posit {
                    bits: bits & Self::bits_mask(),
                    _marker: PhantomData,
                }
            }

            /// Computes the absolute value of `self`
            pub fn abs(self) -> Self {
                if self.is_negative() {
                    -self
                } else {
                    self
                }
            }

            /// Returns the bit representation of `self`
            pub fn bits(self) -> $bits {
                self.bits
            }

            /// Returns the largest integer less than or equal to `self`
            #[cfg(unimplemented)]
            pub fn ceil(self) -> Self {}

            /// Returns the exponent component of `self`
            pub fn exponent(self) -> $bits {
                (self.abs().bits >> self.fraction_size()) & self.exponent_mask()
            }

            /// Returns size, in bits, of the exponent component of `self`
            pub fn exponent_size(self) -> u8 {
                cmp::min(Self::nbits() - Self::sign_size() - self.regime_size(),
                         ES::to_u8())
            }

            /// Returns the smallest integer greater than or equal to `self`
            #[cfg(unimplemented)]
            pub fn floor(self) -> Self {}

            /// Returns the fraction component of `self`
            pub fn fraction(self) -> $bits {
                self.abs().bits & self.fraction_mask()
            }

            /// Returns size, in bits, of the fraction component of `self`
            pub fn fraction_size(self) -> u8 {
                Self::nbits() - Self::sign_size() - self.regime_size() -
                    self.exponent_size()
            }

            /// Largest positive value that this posit can hold
            pub fn largest() -> Self {
                Posit { bits: (1 << Self::nbits() - 1), _marker: PhantomData }
            }

            /// Returns `true` if this value is not infinity
            pub fn is_finite(self) -> bool {
                !self.is_infinity()
            }

            /// Returns `true` if this value is infinity
            pub fn is_infinity(self) -> bool {
                self.bits == Self::sign_mask()
            }

            /// Returns `true` if this value is negative AND not infinity
            pub fn is_negative(self) -> bool {
                self.bits & Self::sign_mask() != 0 && self.is_finite()
            }

            /// Returns the reciprocal (inverse) of `self`, i.e. `1 / self`
            pub fn recip(mut self) -> Self {
                // Unums 2.0 says "To reciprocate reverse all bits but the
                // first one and add one", but this only seems to be true for
                // exact posits, i.e. posits with fractional part equal to zero
                if self.fraction() == 0 {
                    self.bits = (self.bits ^
                                 (Self::bits_mask() & !Self::sign_mask())) + 1;
                } else {
                    unimplemented!()
                }

                self
            }

            /// Returns the size, in bits, of the regime component
            pub fn regime_size(self) -> u8 {
                // left aligned bits
                let bits = self.abs().bits <<
                    (Self::bits_size() - Self::nbits());

                // chop off the sign bit
                let bits = bits << Self::sign_size();

                let lz = bits.leading_zeros() as u8;
                let lo = (!bits).leading_zeros() as u8;
                let rs = cmp::max(lz, lo) + 1;

                cmp::min(rs, Self::nbits() - 1)
            }

            /// Returns the nearest integer to `self`
            #[cfg(unimplemented)]
            pub fn round(self) -> Self {}

            /// Returns the sign bit
            pub fn sign(self) -> $bits {
                (self.bits & Self::sign_mask())
                    >> (Self::nbits() - Self::sign_size())
            }

            /// Smallest positive value that this posit can hold
            pub fn smallest() -> Self {
                Posit { bits: 1, _marker: PhantomData }
            }


            /// Returns the integer part of `self`
            #[cfg(unimplemented)]
            pub fn trunc(mut self) -> Self {}

            fn bits_mask() -> $bits {
                (1 as $bits)
                    .checked_shl(u32(Self::nbits()))
                    .map(|x| x - 1)
                    .unwrap_or($bits::MAX)
            }

            /// Size of `$bits`
            fn bits_size() -> u8 {
                $bits_size
            }

            fn exponent_mask(self) -> $bits {
                (1 << self.exponent_size()) - 1
            }

            fn fraction_mask(self) -> $bits {
                (1 << self.fraction_size()) - 1
            }

            fn infinity() -> Self {
                Posit { bits: 1 << (Self::nbits() - 1), _marker: PhantomData }
            }

            fn is_one(self) -> bool {
                self.abs().bits == (1 << (Self::nbits() - 2))
            }

            fn is_zero(self) -> bool {
                self.bits == 0
            }

            /// Number of bits that the posit uses
            fn nbits() -> u8 {
                NBITS::to_u8()
            }

            #[cfg(unused)]
            fn one() -> Self {
                Posit { bits: (1 << (Self::nbits() - 2)), _marker: PhantomData }
            }

            /// Returns the regime as an exponent
            fn regime(self) -> i8 {
                // left aligned bits
                let bits = self.abs().bits <<
                    (Self::bits_size() - Self::nbits());

                // chop off the sign bit
                let bits = bits << Self::sign_size();

                let lz = bits.leading_zeros() as i8;
                let lo = (!bits).leading_zeros() as i8;

                if lz == 0 { lo - 1 } else { -lz }
            }

            fn sign_mask() -> $bits {
                1 << (Self::nbits() - Self::sign_size())
            }

            fn sign_size() -> u8 {
                1
            }

            fn zero() -> Self {
                Posit { bits: 0, _marker: PhantomData }
            }
        }

        impl<NBITS, ES> PartialOrd for Posit<$bits, NBITS, ES>
        where
            ES: Unsigned,
            NBITS: Unsigned,
        {
            fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
                // infinity has no sign information so we can't make any
                // conclusion
                if self.is_infinity() || rhs.is_infinity() {
                    None
                } else {
                    // The rest of posits are ordered along the real number line
                    Some(self.bits.cmp(&rhs.bits))
                }
            }
        }

        impl<NBITS, ES> cast::From<Posit<$bits, NBITS, ES>> for f64
        where
            ES: Unsigned,
            NBITS: Unsigned,
        {
            type Output = f64;

            fn cast(p: Posit<$bits, NBITS, ES>) -> Self::Output {
                if p.is_zero() {
                    return 0.
                } else if p.is_infinity() {
                    return 1. / 0.
                }

                let bits_size = 64;
                let sign_size = 1;
                let exponent_size = 11;
                let exponent_bias = (1i32 << (exponent_size - 1)) - 1;
                let fraction_size = bits_size - sign_size - exponent_size;

                // XXX(i32) Here we assume that ES < 32
                let posit_exponent = (1 << ES::to_u8()) * i32(p.regime()) +
                    p.exponent() as i32;

                let (exponent, fraction_bits) =
                    if let Ok(exponent) = u64(posit_exponent + exponent_bias) {
                        if exponent > 1u64 << exponent_size {
                            // overflow
                            ((1 << exponent_size) - 1, (1 << fraction_size) - 1)
                        } else {
                            (exponent,
                             u64(p.fraction()) <<
                             (fraction_size - p.fraction_size()))
                        }
                    } else {
                        // underflow
                        (0, 0)
                    };

                let sign_bits = u64(p.sign()) << (bits_size - sign_size);
                let exponent_bits = exponent << fraction_size;

                unsafe {
                    mem::transmute(sign_bits |
                                   exponent_bits |
                                   fraction_bits)
                }
            }
        }

        impl<NBITS, ES> cast::From<f64> for Posit<$bits, NBITS, ES>
            where
            ES: Unsigned,
            NBITS: Unsigned,
        {
            type Output = Result<Posit<$bits, NBITS, ES>, Error>;

            fn cast(x: f64) -> Self::Output {
                if x != x {
                    return Err(Error::NaN);
                } else if x == 0. {
                    return Ok(Posit::<$bits, NBITS, ES>::zero());
                } else if x == f64::INFINITY || x == f64::NEG_INFINITY {
                    return Ok(Posit::<$bits, NBITS, ES>::infinity());
                }

                let nbits = Self::nbits();
                let bits_size = 64;
                let sign_size = 1;
                let exponent_size = 11;
                let exponent_bias = (1i16 << (exponent_size - 1)) - 1;
                let ieee754_fraction_size = bits_size -
                    sign_size - exponent_size;

                let exponent_mask = (1 << exponent_size) - 1;
                let fraction_mask = (1 << ieee754_fraction_size) - 1;

                let bits: u64 = unsafe { mem::transmute(x) };
                let ieee754_exponent = ((bits >> ieee754_fraction_size) &
                                        exponent_mask) as i16 - exponent_bias;
                let ieee754_fraction = bits & fraction_mask;

                let max_exponent = 2 * (1 << ES::to_u8()) * (i16(nbits) - 2);

                // clamp exponent
                let posit_exponent = cmp::min(cmp::max(ieee754_exponent,
                                                       -max_exponent),
                                              max_exponent);

                let sign = (bits >> (bits_size - sign_size)) as $bits;

                let regime = posit_exponent >> ES::to_u8();
                let exponent = (posit_exponent -
                                regime * (1 << ES::to_u8())) as $bits;

                let (regime_bits, regime_size) = if regime < 0 {
                    (1, (1 - regime) as u8)
                } else if regime == i16(nbits) - 2 {
                    ((1 << (nbits - 1)) - 1, nbits - 1)
                } else {
                    ((1 << (regime + 2)) - 2, regime as u8 + 2)
                };

                let exponent_size = cmp::min(nbits -
                                             Self::sign_size() - regime_size,
                                             ES::to_u8());
                let exponent_mask = (1 << exponent_size) - 1;

                let fraction_size = nbits - Self::sign_size() -
                    regime_size - exponent_size;

                let mut bits = regime_bits << (nbits - Self::sign_size() -
                                               regime_size);
                bits |= (exponent & exponent_mask) <<
                    (nbits - Self::sign_size() -
                     regime_size - exponent_size);
                bits |= (ieee754_fraction >> (ieee754_fraction_size -
                                              fraction_size)) as $bits;

                let p = Posit { bits: bits, _marker: PhantomData };

                if sign == 1 {
                    Ok(-p)
                } else {
                    Ok(p)
                }
            }
        }

        impl<NBITS, ES> fmt::Debug for Posit<$bits, NBITS, ES>
        where
            ES: Unsigned,
            NBITS: Unsigned,
        {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f
                    .debug_struct("Posit")
                    .field("bits", &self.bits)
                    .field("nbits", &NBITS::to_u8())
                    .field("sign", &self.sign())
                    .field("regime", &self.regime())
                    .field("rs", &self.regime_size())
                    .field("exponent", &self.exponent())
                    .field("es", &self.exponent_size())
                    .field("fraction", &self.fraction())
                    .field("fs", &self.fraction_size())
                    .finish()
            }
        }

        impl<NBITS, ES> fmt::Display for Posit<$bits, NBITS, ES>
        where
            ES: Unsigned,
            NBITS: Unsigned,
        {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                cast::f64(*self).fmt(f)
            }
        }

        impl<NBITS, ES> ops::Add for Posit<$bits, NBITS, ES>
            where
            ES: Unsigned,
            NBITS: Unsigned,
        {
            type Output = Self;

            fn add(self, rhs: Self) -> Self {
                if self.is_zero() {
                    return rhs;
                } else if rhs.is_zero() {
                    return self;
                } else if self.is_infinity() && rhs.is_infinity() {
                    panic!("NaN")
                } else if self.is_infinity() || rhs.is_infinity() {
                    return Self::infinity();
                } else if self == -rhs {
                    return Self::zero();
                }

                let es = ES::to_u8();
                let nbits = Self::nbits();

                // XXX(i32) here we assume ES < 32
                let lhs_e = i32(self.regime()) * (1 << es) +
                    self.exponent() as i32;
                let rhs_e = i32(rhs.regime()) * (1 << es) +
                    rhs.exponent() as i32;

                let lhs_sign = self.sign();
                let lhs_fs = self.fraction_size();
                // add hidden bit
                let lhs_f = self.fraction() | (1 << lhs_fs);

                let rhs_sign = rhs.sign();
                let rhs_fs = rhs.fraction_size();
                // add hidden bit
                let rhs_f = rhs.fraction() | (1 << rhs_fs);

                let (s_sign, mut s_f, mut s_e, s_fs) = if lhs_e == rhs_e {
                    let (lhs, rhs, fs) = if lhs_fs > rhs_fs {
                        let lhs = lhs_f;
                        let rhs = (rhs_f << (lhs_fs - rhs_fs)) >> (lhs_e - rhs_e);

                        (lhs, rhs, lhs_fs)
                    } else {
                        let lhs = lhs_f << (rhs_fs - lhs_fs);
                        let rhs = rhs_f >> (lhs_e - rhs_e);

                        (lhs, rhs, rhs_fs)
                    };

                    let (sign, s) = if lhs_sign ^ rhs_sign == 0 {
                        // same sign
                        (lhs_sign, lhs + rhs)
                    } else {
                        if lhs > rhs {
                            (lhs_sign, lhs - rhs)
                        } else {
                            (rhs_sign, rhs - lhs)
                        }
                    };

                    (sign, s, lhs_e, fs)
                } else if lhs_e > rhs_e {
                    let (lhs, rhs, fs) = if lhs_fs > rhs_fs {
                        let lhs = lhs_f;
                        let rhs = (rhs_f << (lhs_fs - rhs_fs)) >> (lhs_e - rhs_e);

                        (lhs, rhs, lhs_fs)
                    } else {
                        let lhs = lhs_f << (rhs_fs - lhs_fs);
                        let rhs = rhs_f >> (lhs_e - rhs_e);

                        (lhs, rhs, rhs_fs)
                    };

                    let s = if lhs_sign ^ rhs_sign == 0 {
                        // same sign
                        lhs + rhs
                    } else {
                        lhs - rhs
                    };

                    (lhs_sign, s, lhs_e, fs)
                } else {
                    // rhs_e > lhs_e
                    let (lhs, rhs, fs) = if lhs_fs > rhs_fs {
                        let lhs = lhs_f >> (rhs_e - lhs_e);
                        let rhs = rhs_f << (lhs_fs - rhs_fs);

                        (lhs, rhs, lhs_fs)
                    } else {
                        let lhs = (lhs_f << (rhs_fs - lhs_fs)) >> (rhs_e - lhs_e);
                        let rhs = rhs_f;

                        (lhs, rhs, rhs_fs)
                    };

                    let s = if lhs_sign ^ rhs_sign == 0 {
                        // same sign
                        rhs + lhs
                    } else {
                        rhs - lhs
                    };

                    (rhs_sign, s, rhs_e, fs)
                };

                // panic!("{:#?}", (self, rhs, s_sign, s_f));

                // adjust fraction and exponent
                if s_f >= (1 << (s_fs + 1)) {
                    s_f >>= 1;
                    s_e += 1
                } else {
                    while s_f < (1 << s_fs) {
                        s_f <<= 1;
                        s_e -= 1;
                    }
                }

                // check for underflow / overflow
                let max_exponent = 2 * (i32(nbits) - 2) * (1 << es);
                if s_e > max_exponent {
                    if s_sign == 1 {
                        return -Self::largest()
                    } else {
                        return Self::largest()
                    }
                } else if s_e < -max_exponent {
                    if s_sign == 1 {
                        return -Self::smallest()
                    } else {
                        return Self::smallest()
                    }
                }

                let regime = (s_e >> es) as i8;
                let exponent = (s_e - i32(regime) * (1 << es)) as $bits;

                let regime_size = cmp::max(-regime + 1, regime + 2) as u8;
                let regime_mask = (1 << regime_size) - 1;

                let regime_bits = if regime < 0 {
                    // A negative regime starts with zeros and ends with a 1.
                    1
                } else {
                    // A positive regime starts with ones and ends with a 0.
                    (Self::bits_mask() << 1) & regime_mask
                };

                let exponent_size = cmp::min(nbits -
                                             Self::sign_size() - regime_size,
                                             es);
                let exponent_mask = (1 << exponent_size) - 1;

                let fraction_size = nbits - Self::sign_size() -
                    regime_size - exponent_size;

                let mut bits = regime_bits << (nbits - Self::sign_size() -
                                               regime_size);

                bits |= (exponent & exponent_mask) <<
                    (nbits - Self::sign_size() -
                     regime_size - exponent_size);

                // here we remove the hidden bit
                bits |= if fraction_size > s_fs {
                    (s_f << (fraction_size - s_fs)) & !(1 << s_fs)
                } else {
                    (s_f >> (s_fs - fraction_size)) & !(1 << fraction_size)
                };

                let p = Posit { bits: bits, _marker: PhantomData };

                if s_sign == 1 {
                    -p
                } else {
                    p
                }
            }
        }

        impl<NBITS, ES> ops::Div for Posit<$bits, NBITS, ES>
            where
            ES: Unsigned,
            NBITS: Unsigned,
        {
            type Output = Self;

            fn div(self, rhs: Self) -> Self {
                self * rhs.recip()
            }
        }

        impl<NBITS, ES> ops::Mul for Posit<$bits, NBITS, ES>
            where
            ES: Unsigned,
            NBITS: Unsigned,
        {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self {
                // fast paths
                if self.is_zero() {
                    if rhs.is_infinity() {
                        panic!("NaN")
                    } else {
                        return Self::zero()
                    }
                } else if rhs.is_zero() {
                    if self.is_infinity() {
                        panic!("NaN")
                    } else {
                        return Self::zero()
                    }
                } else if self.is_one() {
                    if self.is_negative() {
                        return -rhs;
                    } else {
                        return rhs;
                    }
                } else if rhs.is_one() {
                    if rhs.is_negative() {
                        return -self;
                    } else {
                        return self;
                    }
                } else if self.is_infinity() || rhs.is_infinity() {
                    return Self::infinity();
                }
                // XXX This may be too expensive
                // } else if self == rhs.recip() {
                //     return Self::one();
                // } else if self == -rhs.recip() {
                //     return -Self::one();
                // }

                let es = ES::to_u8();
                let nbits = Self::nbits();

                // XXX(i32) here we assume ES < 32
                let lhs_e = i32(self.regime()) * (1 << es) +
                    self.exponent() as i32;
                let rhs_e = i32(rhs.regime()) * (1 << es) +
                    rhs.exponent() as i32;

                let lhs_fs = self.fraction_size();
                let rhs_fs = rhs.fraction_size();

                // fraction + hidden bit
                let lhs_f = self.fraction() | (1 << lhs_fs);
                let rhs_f = rhs.fraction() | (1 << rhs_fs);

                // product of fractions: 1.abc * 1.def
                let p_f = $dbits(lhs_f) * $dbits(rhs_f);

                // whether the product exceeds 2.0
                let carry = (p_f >> (lhs_fs + rhs_fs + 1)) == 1;

                // XXX we should probably return maxpos / minpos at this point
                // clip exponent
                let max_exponent = 2 * (i32(nbits) - 2) * (1 << es);
                let e = cmp::min(cmp::max(lhs_e + rhs_e + carry as i32,
                                          -max_exponent),
                                 max_exponent);

                let sign = self.sign() ^ rhs.sign();
                let regime = (e >> es) as i8;
                let exponent = (e - i32(regime) * (1 << es)) as $bits;

                let regime_size = cmp::max(-regime + 1, regime + 2) as u8;
                let regime_mask = (1 << regime_size) - 1;

                let regime_bits = if regime < 0 {
                    1
                } else {
                    let bits_mask = Self::bits_mask();
                    (bits_mask & (bits_mask << (regime + 1))) & regime_mask
                };

                let exponent_size = cmp::min(nbits -
                                             Self::sign_size() - regime_size,
                                             es);
                let exponent_mask = (1 << exponent_size) - 1;

                let fraction_size = nbits - Self::sign_size() -
                    regime_size - exponent_size;

                let mut bits = regime_bits << (nbits - Self::sign_size() -
                                               regime_size);
                bits |= (exponent & exponent_mask) <<
                    (nbits - Self::sign_size() -
                     regime_size - exponent_size);

                // here we remove the hidden bit
                bits |= ((p_f >> carry as u8)
                         >> (lhs_fs + rhs_fs - fraction_size)) as $bits &
                    !(1 << fraction_size);

                let p = Posit { bits: bits, _marker: PhantomData };

                if sign == 1 {
                    -p
                } else {
                    p
                }
            }
        }

        impl<NBITS, ES> ops::Sub for Posit<$bits, NBITS, ES>
            where
            ES: Unsigned,
            NBITS: Unsigned,
        {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self {
                self + (-rhs)
            }
        }

        impl<NBITS, ES> ops::Neg for Posit<$bits, NBITS, ES>
            where
            ES: Unsigned,
            NBITS: Unsigned,
        {
            type Output = Self;

            fn neg(mut self) -> Self {
                // Negation is the 2's complement of the bit pattern
                // i.e. reverse all bits and add one
                self.bits = (self.bits ^ Self::bits_mask()) + 1;
                self
            }
        }
    }
}

posit!(u8, u16, U9, 8);
posit!(u16, u32, U17, 16);
posit!(u32, u64, U33, 32);

// const fn new
#[cfg(feature = "const-fn")]
macro_rules! new {
    ($nbits:ident, $bits:ident, $mask:expr) => {
        impl<ES> Posit<$bits, $nbits, ES>
        where
            ES: Unsigned + Cmp<$nbits, Output = Less>,
        {
            /// Creates a new posit from the bit pattern `bits`
            pub const fn new(bits: $bits) -> Self {
                Posit {
                    bits: bits & $mask,
                    _marker: PhantomData,
                }
            }
        }
    }
}

#[cfg(not(feature = "const-fn"))]
macro_rules! new {
    ($($tt:tt)+) => {}
}

new!(U3, u8, 0b111);
new!(U4, u8, 0b1111);
new!(U5, u8, 0b1_1111);
new!(U6, u8, 0b11_1111);
new!(U7, u8, 0b111_1111);
new!(U8, u8, 0b1111_1111);

new!(U9, u16, 0b1_1111_1111);
new!(U10, u16, 0b11_1111_1111);
new!(U11, u16, 0b111_1111_1111);
new!(U12, u16, 0b1111_1111_1111);
new!(U13, u16, 0b1_1111_1111_1111);
new!(U14, u16, 0b11_1111_1111_1111);
new!(U15, u16, 0b111_1111_1111_1111);
new!(U16, u16, 0b1111_1111_1111_1111);

new!(U17, u32, 0b1_1111_1111_1111_1111);
new!(U18, u32, 0b11_1111_1111_1111_1111);
new!(U19, u32, 0b111_1111_1111_1111_1111);
new!(U20, u32, 0b1111_1111_1111_1111_1111);
new!(U21, u32, 0b1_1111_1111_1111_1111_1111);
new!(U22, u32, 0b11_1111_1111_1111_1111_1111);
new!(U23, u32, 0b111_1111_1111_1111_1111_1111);
new!(U24, u32, 0b1111_1111_1111_1111_1111_1111);
new!(U25, u32, 0b1_1111_1111_1111_1111_1111_1111);
new!(U26, u32, 0b11_1111_1111_1111_1111_1111_1111);
new!(U27, u32, 0b111_1111_1111_1111_1111_1111_1111);
new!(U28, u32, 0b1111_1111_1111_1111_1111_1111_1111);
new!(U29, u32, 0b1_1111_1111_1111_1111_1111_1111_1111);
new!(U30, u32, 0b11_1111_1111_1111_1111_1111_1111_1111);
new!(U31, u32, 0b111_1111_1111_1111_1111_1111_1111_1111);
new!(U32, u32, 0b1111_1111_1111_1111_1111_1111_1111_1111);

// Type alias and checked cast function
macro_rules! ty {
    ($ty:ident, $bits:ident, $nbits:ident, $es:ident) => {
        pub type $ty = Posit<$bits, $nbits, $es>;

        #[allow(non_snake_case)]
        pub fn $ty<T>(x: T) -> <$ty as cast::From<T>>::Output
            where $ty: cast::From<T>,
        {
            <$ty as cast::From<T>>::cast(x)
        }
    }
}

ty!(P8E0, u8, U8, U0);
ty!(P8E1, u8, U8, U1);
ty!(P8E2, u8, U8, U2);
ty!(P8E3, u8, U8, U3);
ty!(P8E4, u8, U8, U4);

ty!(P16E0, u16, U16, U0);
ty!(P16E1, u16, U16, U1);
ty!(P16E2, u16, U16, U2);
ty!(P16E3, u16, U16, U3);
ty!(P16E4, u16, U16, U4);
ty!(P16E5, u16, U16, U5);

ty!(P32E0, u32, U32, U0);
ty!(P32E1, u32, U32, U1);
ty!(P32E2, u32, U32, U2);
ty!(P32E3, u32, U32, U3);
ty!(P32E4, u32, U32, U4);
ty!(P32E5, u32, U32, U5);
ty!(P32E6, u32, U32, U6);
ty!(P32E7, u32, U32, U7);
ty!(P32E8, u32, U32, U8);

#[cfg(test)]
mod tests {
    use cast::{Error, From, f64};
    use quickcheck::TestResult;
    use typenum::{U1, U3, U4, U5};

    use super::Posit;

    type P3E1 = Posit<u8, U3, U1>;

    #[test]
    fn p3e1() {
        assert_eq!(f64(P3E1::new(0b000)), 0.);
        assert_eq!(f64(P3E1::new(0b001)), 1. / 4.);
        assert_eq!(f64(P3E1::new(0b010)), 1.);
        assert_eq!(f64(P3E1::new(0b011)), 4.);
        assert_eq!(f64(P3E1::new(0b100)), 1. / 0.);
        assert_eq!(f64(P3E1::new(0b101)), -4.);
        assert_eq!(f64(P3E1::new(0b110)), -1.);
        assert_eq!(f64(P3E1::new(0b111)), -1. / 4.);

        for &x in &[0., 1. / 4., 1., 4., 1. / 0., -4., -1., -1. / 4.] {
            assert_eq!(f64(P3E1::cast(x).unwrap()), x);
        }
    }

    type P4E1 = Posit<u8, U4, U1>;

    #[test]
    fn p4e1() {
        assert_eq!(f64(P4E1::new(0b0000)), 0.);
        assert_eq!(f64(P4E1::new(0b0001)), 1. / 16.);
        assert_eq!(f64(P4E1::new(0b0010)), 1. / 4.);
        assert_eq!(f64(P4E1::new(0b0011)), 1. / 2.);
        assert_eq!(f64(P4E1::new(0b0100)), 1.);
        assert_eq!(f64(P4E1::new(0b0101)), 2.);
        assert_eq!(f64(P4E1::new(0b0110)), 4.);
        assert_eq!(f64(P4E1::new(0b0111)), 16.);
        assert_eq!(f64(P4E1::new(0b1000)), 1. / 0.);
        assert_eq!(f64(P4E1::new(0b1001)), -16.);
        assert_eq!(f64(P4E1::new(0b1010)), -4.);
        assert_eq!(f64(P4E1::new(0b1011)), -2.);
        assert_eq!(f64(P4E1::new(0b1100)), -1.);
        assert_eq!(f64(P4E1::new(0b1101)), -1. / 2.);
        assert_eq!(f64(P4E1::new(0b1110)), -1. / 4.);
        assert_eq!(f64(P4E1::new(0b1111)), -1. / 16.);

        for &x in &[
            0.,
            1. / 16.,
            1. / 4.,
            1. / 2.,
            1.,
            2.,
            4.,
            16.,
            1. / 0.,
            -16.,
            -4.,
            -2.,
            -1.,
            -1. / 2.,
            -1. / 4.,
            -1. / 16.,
        ] {
            assert_eq!(f64(P4E1::cast(x).unwrap()), x);
        }
    }

    type P5E1 = Posit<u8, U5, U1>;

    #[test]
    fn p5e1() {
        assert_eq!(f64(P5E1::new(0b00000)), 0.);
        assert_eq!(f64(P5E1::new(0b00001)), 1. / 64.);
        assert_eq!(f64(P5E1::new(0b00010)), 1. / 16.);
        assert_eq!(f64(P5E1::new(0b00011)), 1. / 8.);
        assert_eq!(f64(P5E1::new(0b00100)), 1. / 4.);
        assert_eq!(f64(P5E1::new(0b00101)), 3. / 8.);
        assert_eq!(f64(P5E1::new(0b00110)), 1. / 2.);
        assert_eq!(f64(P5E1::new(0b00111)), 3. / 4.);
        assert_eq!(f64(P5E1::new(0b01000)), 1.);
        assert_eq!(f64(P5E1::new(0b01001)), 3. / 2.);
        assert_eq!(f64(P5E1::new(0b01010)), 2.);
        assert_eq!(f64(P5E1::new(0b01011)), 3.);
        assert_eq!(f64(P5E1::new(0b01100)), 4.);
        assert_eq!(f64(P5E1::new(0b01101)), 8.);
        assert_eq!(f64(P5E1::new(0b01110)), 16.);
        assert_eq!(f64(P5E1::new(0b01111)), 64.);
        assert_eq!(f64(P5E1::new(0b10000)), 1. / 0.);
        assert_eq!(f64(P5E1::new(0b10001)), -64.);
        assert_eq!(f64(P5E1::new(0b10010)), -16.);
        assert_eq!(f64(P5E1::new(0b10011)), -8.);
        assert_eq!(f64(P5E1::new(0b10100)), -4.);
        assert_eq!(f64(P5E1::new(0b10101)), -3.);
        assert_eq!(f64(P5E1::new(0b10110)), -2.);
        assert_eq!(f64(P5E1::new(0b10111)), -3. / 2.);
        assert_eq!(f64(P5E1::new(0b11000)), -1.);
        assert_eq!(f64(P5E1::new(0b11001)), -3. / 4.);
        assert_eq!(f64(P5E1::new(0b11010)), -1. / 2.);
        assert_eq!(f64(P5E1::new(0b11011)), -3. / 8.);
        assert_eq!(f64(P5E1::new(0b11100)), -1. / 4.);
        assert_eq!(f64(P5E1::new(0b11101)), -1. / 8.);
        assert_eq!(f64(P5E1::new(0b11110)), -1. / 16.);
        assert_eq!(f64(P5E1::new(0b11111)), -1. / 64.);

        for &x in &[
            0.,
            1. / 64.,
            1. / 16.,
            1. / 8.,
            1. / 4.,
            3. / 8.,
            1. / 2.,
            3. / 4.,
            1.,
            3. / 2.,
            2.,
            3.,
            4.,
            8.,
            16.,
            64.,
            1. / 0.,
            -64.,
            -16.,
            -8.,
            -4.,
            -3.,
            -2.,
            -3. / 2.,
            -1.,
            -3. / 4.,
            -1. / 2.,
            -3. / 8.,
            -1. / 4.,
            -1. / 8.,
            -1. / 16.,
            -1. / 64.,
        ] {
            assert_eq!(f64(P5E1::cast(x).unwrap()), x);
        }
    }

    use super::P32E6;

    quickcheck! {
        fn add(x: f64, y: f64) -> TestResult {
            (|| -> Result<_, Error> {
                let ans = P32E6(x + y)?;
                let s = P32E6(x)? + P32E6(y)?;

                Ok(s == ans ||
                   relative_eq!(x + y, f64(s),
                                epsilon = 1e-5, max_relative = (1. + 1e-5)))
            })().map(TestResult::from_bool).unwrap_or(TestResult::discard())
        }

        fn commutative_add(x: f64, y: f64) -> TestResult {
            (|| -> Result<_, Error> {
                let x = P32E6(x)?;
                let y = P32E6(y)?;

                Ok((x + y) == (y + x))
            })().map(TestResult::from_bool).unwrap_or(TestResult::discard())
        }

        fn commutative_mul(x: f64, y: f64) -> TestResult {
            (|| -> Result<_, Error> {
                let x = P32E6(x).unwrap();
                let y = P32E6(y).unwrap();

                Ok((x * y) == (y * x))
            })().map(TestResult::from_bool).unwrap_or(TestResult::discard())
        }

        // TODO `recip` is not complete
        #[ignore]
        fn div(x: f64, y: f64) -> TestResult {
            (|| -> Result<_, Error> {
                let ans = P32E6(x / y)?;
                let p = P32E6(x)? / P32E6(y)?;

                Ok(p == ans ||
                   relative_eq!(x / y, f64(p),
                                epsilon = 1e-5, max_relative = (1. + 1e-5)))
            })().map(TestResult::from_bool).unwrap_or(TestResult::discard())
        }

        fn mul(x: f64, y: f64) -> TestResult {
            (|| -> Result<_, Error> {
                let ans = P32E6(x * y).unwrap();
                let p = P32E6(x).unwrap() * P32E6(y).unwrap();

                Ok(p == ans ||
                   relative_eq!(x * y, f64(p),
                                epsilon = 1e-5, max_relative = (1. + 1e-5)))
            })().map(TestResult::from_bool).unwrap_or(TestResult::discard())
        }

        fn sub(x: f64, y: f64) -> TestResult {
            (|| -> Result<_, Error> {
                let ans = P32E6(x - y)?;
                let s = P32E6(x)? - P32E6(y)?;

                Ok(s == ans ||
                   relative_eq!(x - y, f64(s),
                                epsilon = 1e-5, max_relative = (1. + 1e-5)))
            })().map(TestResult::from_bool).unwrap_or(TestResult::discard())
        }
    }
}
