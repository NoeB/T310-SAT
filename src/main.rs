mod CCITT2;
mod T310;
mod T310SAT;
use CCITT2::{SimpleCCITT2, SimpleError};

use T310::T310Cipher;
use rand::{Rng, SeedableRng, rngs::StdRng};

pub fn generate_random_key() -> ([bool; 120], [bool; 120]) {
    let mut rng = StdRng::seed_from_u64(123456);

    // Create two arrays with 120 bits each
    let mut s1 = [false; 120];
    let mut s2 = [false; 120];

    // Generate random bits
    for i in 0..120 {
        s1[i] = rng.random_bool(0.5);
        s2[i] = rng.random_bool(0.5);
    }

    // Ensure parity requirement is met (each 24-bit block must have odd parity)
    for i in 0..5 {
        let block_start = i * 24;
        let block_end = block_start + 24;

        // Calculate parity for each block
        let mut s1_parity = false;
        let mut s2_parity = false;

        for j in block_start..block_end {
            s1_parity ^= s1[j];
            s2_parity ^= s2[j];
        }

        // If even parity (false), flip the last bit in the block to make it odd
        if !s1_parity {
            let last_bit_index = block_start + 23;
            s1[last_bit_index] = !s1[last_bit_index];
        }

        if !s2_parity {
            let last_bit_index = block_start + 23;
            s2[last_bit_index] = !s2[last_bit_index];
        }
    }

    (s1, s2)
}

/// Generate a random 61-bit initialization vector for the T-310 cipher
pub fn generate_random_iv() -> [bool; 61] {
    let mut rng = StdRng::seed_from_u64(1234567);

    // Generate a random 64-bit value and mask to 61 bits
    let mut iv = rng.random::<u64>() & 0x1fffffffffffffff;

    // Ensure it's not all zeros
    if iv == 0 {
        iv = 0x1234567890abcdef & 0x1fffffffffffffff
    }

    let mut iv_array = [false; 61];
    for i in 0..61 {
        iv_array[i] = ((iv >> (60 - i)) & 1) == 1;
    }

    iv_array
}

fn main() -> Result<(), SimpleError> {
    let codec = SimpleCCITT2::new();

    // Test encoding and decoding
    //let text = "HELLO 1234 TEST T310 test";
    let text = "HELLO";
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
    let mut cipher = T310Cipher::new(&s1, &s2, &iv);
    let mut cipher_clone = T310Cipher::new(&s1, &s2, &iv);
    let mut encrypted: Vec<bool> = vec![];
    for chunk in bool_array.chunks(5) {
        let chunk_array: [bool; 5] = chunk.try_into().expect("Chunk size must be 5");
        //encrypted.extend_from_slice(&cipher.encrypt_character(chunk_array));
        encrypted.extend_from_slice(&cipher.encrypt_character_simple(chunk_array));
    }
    println!("Encrypted bools: {:?}", &encrypted);
    println!(
        "Encrypted decoded: {:?}",
        codec.decode_from_bools(&encrypted)
    );
    let mut decrypted: Vec<bool> = vec![];
    for chunk in encrypted.chunks(5) {
        let chunk_array: [bool; 5] = chunk.try_into().expect("Chunk size must be 5");
        //decrypted.extend_from_slice(&cipher_clone.decrypt_character(chunk_array));
        decrypted.extend_from_slice(&cipher_clone.encrypt_character_simple(chunk_array));
    }
    println!("Decrypted bools: {:?}", decrypted);

    let decoded_text = codec.decode_from_bools(&decrypted);
    println!("Decoded text: {}", decoded_text.unwrap());

    println!("---------------");
    // Compare SAT output if correect

    let s1_test = [
        false, false, false, false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, true, false, true, false, false, false, true, false, false,
        false, false, true, true, true, true, false, false, true, false, true, true, false, true,
        false, false, true, true, false, true, true, true, true, false, true, false, false, true,
        true, false, false, true, true, false, true, true, false, false, true, true, false, true,
        true, true, true, true, true, true, false, true, true, true, true, true, false, false,
        true, true, true, true, true, true, false, true, false, false, true, true, true, true,
        true, false, true, true, true, false, false, false, true, false, false, false, false, true,
        true, true, true, false, false, false, true, false, false, false,
    ];
    let s2_test: [bool; 120] = [
        false, false, false, false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, true, true, true, true, false, false,
        true, false, false, false, true, true, false, false, true, false, false, true, false, true,
        false, false, false, true, true, true, true, false, true, false, true, false, true, true,
        false, true, false, true, false, false, false, true, true, true, false, false, true, false,
        true, false, true, false, false, false, false, false, false, false, true, true, false,
        true, false, true, true, false, false, false, true, true, false, true, true, false, true,
        false, true, true, true, true, false, true, false, true, false, false, false, false, true,
        true, false, false, true, true, true, true, true, false, false, false,
    ];
    let iv_test = [
        true, false, false, false, true, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false,
    ];

    let mut cipher_clone = T310Cipher::new(&s1_test, &s2_test, &iv_test);

    let mut encrypted: Vec<bool> = vec![];
    for chunk in bool_array.chunks(5) {
        let chunk_array: [bool; 5] = chunk.try_into().expect("Chunk size must be 5");
        //encrypted.extend_from_slice(&cipher.encrypt_character(chunk_array));
        encrypted.extend_from_slice(&cipher_clone.encrypt_character_simple(chunk_array));
    }
    println!("Encrypted bools2: {:?}", &encrypted);
    println!(
        "Encrypted decoded: {:?}",
        codec.decode_from_bools(&encrypted)
    );
    Ok(())
}
