mod CCITT2;

use CCITT2::{SimpleCCITT2, SimpleError};
use bitvec::prelude::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

pub struct T310Cipher {
    // Key components: two 120-bit sequences
    s1: BitVec<u8, Msb0>,
    s2: BitVec<u8, Msb0>,

    // Standard U-Vector: 37-bit register
    u_vector: BitVec<u8, Msb0>,

    // 61-bit synchronization sequence (initialization vector (or F - Vector LFSR))
    f_vector: BitVec<u8, Msb0>,

    //Long Term Key
    // P function mapping {1,2,...27} to {1,...36}
    p: [u8; 27],
    // D function mapping {1,2,...9} to {0,1,...36}
    d: [u8; 9],
    alpha: u8,
    a: [bool; 13],
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

        // Longterm Key LZS-31 approved ()
        let p: [u8; 27] = [
            7, 4, 33, 30, 18, 36, 5, 35, 9, 16, 23, 26, 32, 12, 21, 1, 13, 25, 20, 8, 24, 15, 22,
            29, 10, 28, 6,
        ];
        let d: [u8; 9] = [0, 15, 3, 23, 11, 27, 31, 35, 19];
        let alpha: u8 = 2; // {1,2,3,...36}
        let a = [false; 13];

        println!("s1: {}", s1);
        println!("s2: {}", s2);
        println!("standard_u_vector: {}", standard_u_vector);
        println!("iv_bitvec: {}", iv_bitvec);
        Self {
            s1: s1.clone(),
            s2: s2.clone(),
            u_vector: standard_u_vector,
            f_vector: iv_bitvec,
            p,
            d,
            alpha,
            a,
        }
    }


    #[rustfmt::skip]
    #[allow(dead_code)]
    fn z(&self, e1: bool, e2: bool, e3: bool, e4: bool, e5: bool, e6: bool) -> bool {
        let mut result = e1 ^ e5 ^ e6 ^ (e1 & e4) ^ (e2 & e3) ^ (e2 & e5) ^ (e4 & e5) ^ (e5 & e6);
        result ^= (e1 & e3 & e4) ^ (e1 & e3 & e6) ^ (e1 & e4 & e5) ^ (e2 & e3 & e6) & (e2 & e4 & e6) & (e3 & e5 & e6);
        result ^= (e1 & e2 & e3 & e4) ^(e1 & e2 & e3 & e5) ^ (e1 & e2 & e5 & e6) ^(e2 & e3 & e4 & e6) ^(e1 & e2 & e3 & e4 & e5);
        result ^= (e1 & e3 & e4 & e5 & e6);

        result
    }
    #[allow(dead_code)]
    fn get_f_bit_and_rotate(&mut self) -> bool {
        let new_bit = self.f_vector[0] ^ self.f_vector[1] ^ self.f_vector[2] ^ self.f_vector[5];
        self.f_vector.rotate_right(1);
        self.f_vector.set(0, new_bit);
        return new_bit;
    }

    //get the last bit of the s2 vector
    #[allow(dead_code)]
    fn get_s2_bit(&self) -> bool {
        self.s2[119]
    }
    // get the last bit of the s2 vector
    #[allow(dead_code)]
    fn get_s1_bit(&self) -> bool {
        self.s1[119]
    }

    fn shift_srv(srv: &mut [bool; 5]) {
        let feedback_bit = srv[0] ^ srv[2];
        srv.rotate_left(1);
        srv[4] = feedback_bit;
    }

    pub fn encrypt_character(&mut self, char: [bool; 5]) -> [bool; 5] {
        self.single_round();
        let mut srv_2: [bool; 5] = [self.a[0], self.a[1], self.a[2], self.a[3], self.a[4]];
        let mut srv_3: [bool; 5] = [true; 5];
        while !srv_2.iter().all(|&x| x == true) && !srv_2.iter().all(|&x| x == false) {
            Self::shift_srv(&mut srv_2);
            Self::shift_srv(&mut srv_3);
        }

        srv_2 = srv_3;

        let mut srv_3: [bool; 5] = [self.a[5], self.a[6], self.a[7], self.a[8], self.a[9]];
        for i in 0..5 {
            srv_3[i] ^= char[i];
        }

        while !srv_2.iter().all(|&x| x == true) && !srv_2.iter().all(|&x| x == false) {
            Self::shift_srv(&mut srv_2);
            Self::shift_srv(&mut srv_3);
        }

        srv_3
    }

    pub fn decrypt_character(&mut self, char: [bool; 5]) -> [bool; 5] {
        self.single_round();
        let mut srv_2: [bool; 5] = [self.a[0], self.a[1], self.a[2], self.a[3], self.a[4]];
        let mut srv_3: [bool; 5] = char;
        while !srv_2.iter().all(|&x| x == true) && !srv_2.iter().all(|&x| x == false) {
            Self::shift_srv(&mut srv_2);
            Self::shift_srv(&mut srv_3);
        }
        let keystream: [bool; 5] = [self.a[5], self.a[6], self.a[7], self.a[8], self.a[9]];
        for i in 0..5 {
            srv_3[i] ^= keystream[i];
        }

        srv_3
    }

    fn get_u(&self, p: usize) -> bool {
        if p == 0 {
            return self.get_s1_bit();
        }
        self.u_vector[p]
    }

    #[allow(dead_code)]
    fn single_round(&mut self) {
        let mut t_array = [false; 10];
        for outer_round in 0..12 {
            for inner_round in 0..126 {
                t_array[9] = t_array[8] ^ self.get_u(self.p[28 - 2] as usize - 1);
                t_array[8] = t_array[7]
                    ^ self.z(
                        self.get_u(self.p[22 - 2] as usize - 1),
                        self.get_u(self.p[23 - 2] as usize - 1),
                        self.get_u(self.p[24 - 2] as usize - 1),
                        self.get_u(self.p[25 - 2] as usize - 1),
                        self.get_u(self.p[26 - 2] as usize - 1),
                        self.get_u(self.p[27 - 2] as usize - 1),
                    );
                t_array[7] = (t_array[6] ^ self.get_u(self.p[21 - 2] as usize - 1));
                t_array[6] = t_array[5]
                    ^ self.z(
                        self.get_u(self.p[15 - 2] as usize - 1),
                        self.get_u(self.p[16 - 2] as usize - 1),
                        self.get_u(self.p[17 - 2] as usize - 1),
                        self.get_u(self.p[18 - 2] as usize - 1),
                        self.get_u(self.p[19 - 2] as usize - 1),
                        self.get_u(self.p[20 - 2] as usize - 1),
                    );
                t_array[5] = t_array[4] ^ self.get_u(self.p[14 - 2] as usize - 1);
                t_array[4] = t_array[3]
                    ^ self.z(
                        self.get_u(self.p[8 - 2] as usize - 1),
                        self.get_u(self.p[9 - 2] as usize - 1),
                        self.get_u(self.p[10 - 2] as usize - 1),
                        self.get_u(self.p[11 - 2] as usize - 1),
                        self.get_u(self.p[12 - 2] as usize - 1),
                        self.get_u(self.p[13 - 2] as usize - 1),
                    );
                t_array[3] = t_array[2] ^ self.get_u(self.p[7 - 2] as usize - 1);
                t_array[2] = t_array[1]
                    ^ self.z(
                        self.get_s2_bit(),
                        self.get_u(self.p[2 - 2] as usize - 1),
                        self.get_u(self.p[3 - 2] as usize),
                        self.get_u(self.p[4 - 2] as usize),
                        self.get_u(self.p[5 - 2] as usize),
                        self.get_u(self.p[6 - 2] as usize - 1),
                    );
                t_array[1] = self.get_f_bit_and_rotate();

                // Shift U vector right by one and insert 9 bits from T into U
                self.u_vector.rotate_right(1);
                let old_u = self.u_vector.clone();
                for j in 1..10 {
                    self.u_vector
                        .set(4 * j - 3, old_u[self.d[j - 1] as usize] ^ t_array[10 - j]);
                }
                let s1_bit = self.get_s1_bit();
                self.u_vector.set(0, s1_bit);
                // Shift s1 and s2
                self.s1.rotate_right(1);
                self.s2.rotate_right(1);
            }
            self.a[outer_round] = self.u_vector[self.alpha as usize - 1]
        }
    }
}

pub fn generate_random_key() -> (BitVec<u8, Msb0>, BitVec<u8, Msb0>) {
    let mut rng = StdRng::seed_from_u64(123456);

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
    let mut rng = StdRng::seed_from_u64(1234567);

    // Generate a random 64-bit value and mask to 61 bits
    let iv = rng.random::<u64>() & 0x1fffffffffffffff;

    // Ensure it's not all zeros
    if iv == 0 {
        0x1234567890abcdef & 0x1fffffffffffffff
    } else {
        iv
    }
}

fn main() -> Result<(), SimpleError> {
    let codec = SimpleCCITT2::new();

    // Test encoding and decoding
    let text = "HELLO 1234 TEST T310 test";
    println!("Original: {}", text);

    let encoded = codec.encode(text)?;
    println!("Encoded: {}", encoded);

    let decoded = codec.decode(&encoded)?;
    println!("Decoded: {}", decoded);

    // Test bool array
    let bool_array = codec.encode_to_bools(text)?;
    println!("Bool array length: {}", bool_array.len());

    let from_bools = codec.decode_from_bools(&bool_array)?;
    println!("From bools: {}", from_bools);

    let (s1, s2) = generate_random_key();
    let iv = generate_random_iv();

    // Create cipher
    let mut cipher = T310Cipher::new(&s1, &s2, iv);
    let mut cipher_clone = T310Cipher::new(&s1, &s2, iv);
    let mut encrypted: Vec<bool> = vec![];
    for chunk in bool_array.chunks(5) {
        let chunk_array: [bool; 5] = chunk.try_into().expect("Chunk size must be 5");
        encrypted.extend_from_slice(&cipher.encrypt_character(chunk_array));
    }
    println!("Encrypted bools: {:?}", encrypted);
    let mut decrypted: Vec<bool> = vec![];
    for chunk in encrypted.chunks(5) {
        let chunk_array: [bool; 5] = chunk.try_into().expect("Chunk size must be 5");
        decrypted.extend_from_slice(&cipher_clone.decrypt_character(chunk_array));
    }
    println!("Decrypted bools: {:?}", decrypted);

    let decoded_text = codec.decode_from_bools(&decrypted);
    println!("Decoded text: {}", decoded_text.unwrap());

    Ok(())
}
