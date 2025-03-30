use bitvec::prelude::*;
use rand::Rng;

pub struct T310Cipher {
    // Key components: two 120-bit sequences
    s1_bit: BitVec<u8, Msb0>,
    s2_bit: BitVec<u8, Msb0>,

    // Standard U-Vector: 37-bit register
    u_vector: BitVec<u8, Msb0>,

    // 61-bit synchronization sequence (initialization vector)
    synchronfolge: BitVec<u8, Msb0>,

    // Pointer to current position in S1/S2
    s_pointer: usize,
}

impl T310Cipher {
    pub fn new(s1: &BitVec<u8, Msb0>, s2: &BitVec<u8, Msb0>, iv: u64) -> Self {
        // Validate the key components
        assert_eq!(s1.len(), 120, "S1 key must be 120 bits");
        assert_eq!(s2.len(), 120, "S2 key must be 120 bits");
        assert!(iv != 0, "IV must not be all zeros");

        // Convert IV to BitVec
        let mut iv_bitvec = BitVec::<u8, Msb0>::with_capacity(61);
        for i in (0..61).rev() {
            iv_bitvec.push((iv >> i) & 1 == 1);
        }

        // Initialize with the standard U-Vector
        let standard_u_vector = bitvec![u8, Msb0;
            0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1,
            1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1,
            0, 0, 0, 1, 1
        ];

        println!("s1: {}", s1);
        println!("s2: {}", s2);
        println!("standard_u_vector: {}", standard_u_vector);
        println!("iv_bitvec: {}", iv_bitvec);
        Self {
            s1_bit: s1.clone(),
            s2_bit: s2.clone(),
            u_vector: standard_u_vector,
            synchronfolge: iv_bitvec,
            s_pointer: 0,
        }
    }
    #[rustfmt::skip]
    fn z(&self, e1: bool, e2: bool, e3: bool, e4: bool, e5: bool, e6: bool) -> bool {
        let mut result = e1 ^ e5 ^ e6 ^ (e1 & e4) ^ (e2 & e3) ^ (e2 & e5) ^ (e4 & e5) ^ (e5 & e6);
        result ^= (e1 & e3 & e4) ^ (e1 & e3 & e6) ^ (e1 & e4 & e5) ^ (e2 & e3 & e6) & (e2 & e4 & e6) & (e3 & e5 & e6);
        result ^= (e1 & e2 & e3 & e4) ^(e1 & e2 & e3 & e5) ^ (e1 & e2 & e5 & e6) ^(e2 & e3 & e4 & e6) ^(e1 & e2 & e3 & e4 & e5);
        result ^= (e1 & e3 & e4 & e5 & e6);

        result
    }
}

pub fn generate_random_key() -> (BitVec<u8, Msb0>, BitVec<u8, Msb0>) {
    let mut rng = rand::rng();

    // Create two BitVec instances with 120 bits each
    let mut s1 = BitVec::<u8, Msb0>::with_capacity(120);
    let mut s2 = BitVec::<u8, Msb0>::with_capacity(120);

    // Initialize with the right size
    s1.resize(120, false);
    s2.resize(120, false);

    // Generate random bits
    for i in 0..120 {
        s1.set(i, rng.random_bool(0.5));
        s2.set(i, rng.random_bool(0.5));
    }

    // Ensure parity requirement is met (each 24-bit block must have odd parity)
    for i in 0..5 {
        let block_start = i * 24;
        let block_end = block_start + 24;

        // Calculate parity for each block
        let s1_parity = s1[block_start..block_end]
            .iter()
            .fold(false, |acc, bit| acc ^ *bit);
        let s2_parity = s2[block_start..block_end]
            .iter()
            .fold(false, |acc, bit| acc ^ *bit);

        // If even parity (false), flip the last bit in the block to make it odd
        if !s1_parity {
            let last_bit_index = block_start + 23;
            let current_bit = s1[last_bit_index];
            s1.set(last_bit_index, !current_bit);
        }

        if !s2_parity {
            let last_bit_index = block_start + 23;
            let current_bit = s2[last_bit_index];
            s2.set(last_bit_index, !current_bit);
        }
    }

    (s1, s2)
}

/// Generate a random 61-bit initialization vector for the T-310 cipher
pub fn generate_random_iv() -> u64 {
    let mut rng = rand::rng();

    // Generate a random 64-bit value and mask to 61 bits
    let iv = rng.random::<u64>() & 0x1fffffffffffffff;

    // Ensure it's not all zeros
    if iv == 0 {
        0x1234567890abcdef & 0x1fffffffffffffff
    } else {
        iv
    }
}

fn main() {
    let (s1, s2) = generate_random_key();
    let iv = generate_random_iv();

    // Create cipher
    let mut cipher = T310Cipher::new(&s1, &s2, iv);
}
