//! A bit-compatible port of ``numpy.random.default_rng(seed)``.
//!
//! Implements ``SeedSequence``, the ``PCG64`` bit generator, the ziggurat
//! ``standard_normal`` sampler and Lemire bounded integers, so synthetic
//! demo fixtures match the Python implementation draw for draw.

use crate::ziggurat::{FI_BITS, KI, WI_BITS};

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;
const POOL_SIZE: usize = 4;

fn hashmix(mut value: u32, hash_const: &mut u32) -> u32 {
    value ^= *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^= value >> XSHIFT;
    value
}

fn mix(x: u32, y: u32) -> u32 {
    let mut result = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^= result >> XSHIFT;
    result
}

/// ``numpy.random.SeedSequence`` for a non-negative integer entropy.
pub struct SeedSequence {
    pool: [u32; POOL_SIZE],
}

impl SeedSequence {
    pub fn new(entropy: u128) -> Self {
        // _coerce_to_uint32_array: little-endian 32-bit words.
        let mut words = Vec::new();
        let mut value = entropy;
        if value == 0 {
            words.push(0);
        }
        while value > 0 {
            words.push((value & 0xffff_ffff) as u32);
            value >>= 32;
        }
        let mut pool = [0u32; POOL_SIZE];
        let mut hash_const = INIT_A;
        for (i, slot) in pool.iter_mut().enumerate() {
            let source = if i < words.len() { words[i] } else { 0 };
            *slot = hashmix(source, &mut hash_const);
        }
        for i_src in 0..POOL_SIZE {
            for i_dst in 0..POOL_SIZE {
                if i_src != i_dst {
                    let hashed = hashmix(pool[i_src], &mut hash_const);
                    pool[i_dst] = mix(pool[i_dst], hashed);
                }
            }
        }
        for &word in words.iter().skip(POOL_SIZE) {
            for i_dst in 0..POOL_SIZE {
                let hashed = hashmix(word, &mut hash_const);
                pool[i_dst] = mix(pool[i_dst], hashed);
            }
        }
        SeedSequence { pool }
    }

    /// ``generate_state(n_words, np.uint64)``.
    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        let mut hash_const = INIT_B;
        let mut state32 = Vec::with_capacity(n_words * 2);
        for i in 0..n_words * 2 {
            let mut data = self.pool[i % POOL_SIZE];
            data ^= hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            data = data.wrapping_mul(hash_const);
            data ^= data >> XSHIFT;
            state32.push(data);
        }
        (0..n_words)
            .map(|i| state32[2 * i] as u64 | ((state32[2 * i + 1] as u64) << 32))
            .collect()
    }
}

const PCG_MULTIPLIER: u128 = (2_549_297_995_355_413_924u128 << 64) + 4_865_540_595_714_422_341u128;

/// The ``PCG64`` (XSL-RR 128/64) bit generator.
pub struct Pcg64 {
    state: u128,
    inc: u128,
    has_uint32: bool,
    uinteger: u32,
}

impl Pcg64 {
    pub fn from_seed(seed: u128) -> Self {
        let words = SeedSequence::new(seed).generate_state_u64(4);
        let initstate = ((words[0] as u128) << 64) | words[1] as u128;
        let initseq = ((words[2] as u128) << 64) | words[3] as u128;
        let mut generator = Pcg64 {
            state: 0,
            inc: (initseq << 1) | 1,
            has_uint32: false,
            uinteger: 0,
        };
        generator.step();
        generator.state = generator.state.wrapping_add(initstate);
        generator.step();
        generator
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let rot = (self.state >> 122) as u32;
        let value = ((self.state >> 64) as u64) ^ (self.state as u64);
        value.rotate_right(rot)
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let next = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (next >> 32) as u32;
        (next & 0xffff_ffff) as u32
    }

    pub fn next_double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

const ZIGGURAT_NOR_R: f64 = 3.654_152_885_361_009;
const ZIGGURAT_NOR_INV_R: f64 = 0.273_661_237_329_758_3;

/// ``numpy.random.Generator`` subset used by the synthetic fixtures.
pub struct Generator {
    bits: Pcg64,
}

impl Generator {
    /// ``np.random.default_rng(seed)``.
    pub fn default_rng(seed: u64) -> Self {
        Generator {
            bits: Pcg64::from_seed(seed as u128),
        }
    }

    pub fn standard_normal(&mut self) -> f64 {
        loop {
            let mut r = self.bits.next_u64();
            let idx = (r & 0xff) as usize;
            r >>= 8;
            let sign = r & 0x1;
            let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
            let mut x = rabs as f64 * f64::from_bits(WI_BITS[idx]);
            if sign & 0x1 == 1 {
                x = -x;
            }
            if rabs < KI[idx] {
                return x;
            }
            if idx == 0 {
                loop {
                    let xx = -ZIGGURAT_NOR_INV_R * (-self.bits.next_double()).ln_1p();
                    let yy = -(-self.bits.next_double()).ln_1p();
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 0x1 == 1 {
                            -(ZIGGURAT_NOR_R + xx)
                        } else {
                            ZIGGURAT_NOR_R + xx
                        };
                    }
                }
            } else {
                let fi_previous = f64::from_bits(FI_BITS[idx - 1]);
                let fi = f64::from_bits(FI_BITS[idx]);
                if (fi_previous - fi) * self.bits.next_double() + fi < (-0.5 * x * x).exp() {
                    return x;
                }
            }
        }
    }

    /// ``normal(loc, scale, size)`` in C order.
    pub fn normal(&mut self, loc: f64, scale: f64, size: usize) -> Vec<f64> {
        (0..size)
            .map(|_| loc + scale * self.standard_normal())
            .collect()
    }

    fn bounded_lemire_u32(&mut self, rng: u32) -> u32 {
        let rng_excl = rng.wrapping_add(1);
        let mut m = self.bits.next_u32() as u64 * rng_excl as u64;
        let mut leftover = (m & 0xffff_ffff) as u32;
        if leftover < rng_excl {
            let threshold = (u32::MAX - rng) % rng_excl;
            while leftover < threshold {
                m = self.bits.next_u32() as u64 * rng_excl as u64;
                leftover = (m & 0xffff_ffff) as u32;
            }
        }
        (m >> 32) as u32
    }

    /// ``integers(low, high, size)`` for int64 with a 32-bit span.
    pub fn integers(&mut self, low: i64, high: i64, size: usize) -> Vec<i64> {
        let rng = (high - 1 - low) as u64;
        assert!(rng < 0xffff_ffff, "only 32-bit spans are supported");
        if rng == 0 {
            return vec![low; size];
        }
        (0..size)
            .map(|_| low + self.bounded_lemire_u32(rng as u32) as i64)
            .collect()
    }
}
