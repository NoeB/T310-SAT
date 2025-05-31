pub struct T310Cipher {
    // Key components: two 120-bit sequences
    s1: [bool; 120],
    s2: [bool; 120],

    // 61-bit synchronization sequence (initialization vector (or F - Vector LFSR))
    f_vector: [bool; 61],

    //Long Term Key
    // P function mapping {1,2,...27} to {1,...36}
    p: [u8; 27],
    // D function mapping {1,2,...9} to {0,1,...36}
    d: [u8; 9],
    alpha: u8,
    // Standard U-Vector: 36-bit register
    u_vector: [bool; 36],
    //Output Regster
    a: [bool; 13],
}

impl T310Cipher {
    pub fn new(s1: &[bool; 120], s2: &[bool; 120], iv: &[bool; 61]) -> Result<Self, String> {
        // Validate the key components

        //assert!(iv != 0, "IV must not be all zeros");

        // Convert IV to BitVec

        //validate s1 and s2
        for array in [s1, s2] {
            // Check each of the 5 groups of 24 bits
            for i in 0..5 {
                let start_idx = i * 24;
                let end_idx = start_idx + 24;

                // Sum the 24 bits in this group
                let sum: u32 = array[start_idx..end_idx]
                    .iter()
                    .map(|&bit| if bit { 1 } else { 0 })
                    .sum();

                // Check if sum is odd (must equal 1 mod 2)
                if sum % 2 == 0 {
                    return Err("Not valid parity".to_string());
                }
            }
        }

        // Initialize with the standard U-Vector
        let standard_u_vector: [bool; 36] = [
            false, true, true, false, true, false, false, true, true, true, false, false, false,
            true, true, true, true, true, false, false, true, false, false, false, false, true,
            false, true, true, false, true, false, false, false, true, true,
        ];

        // Longterm Key LZS-31 approved (https://scz.bplaced.net/t310-schluessel.html)
        let p: [u8; 27] = [
            7, 4, 33, 30, 18, 36, 5, 35, 9, 16, 23, 26, 32, 12, 21, 1, 13, 25, 20, 8, 24, 15, 22,
            29, 10, 28, 6,
        ];
        let d: [u8; 9] = [0, 15, 3, 23, 11, 27, 31, 35, 19];
        let alpha: u8 = 2; // {1,2,3,...36}
        let a = [false; 13];

        println!("s1: {:?}", s1);
        println!("s2: {:?}", s2);
        /*
        println!("standard_u_vector: {:?}", standard_u_vector);
        println!("iv_bitvec: {:?}", iv);
        */
        Ok(Self {
            s1: s1.clone(),
            s2: s2.clone(),
            u_vector: standard_u_vector,
            f_vector: iv.clone(),
            p,
            d,
            alpha,
            a,
        })
    }

    pub fn get_a(&self) -> [bool; 13] {
        self.a.clone()
    }
    #[rustfmt::skip]
    #[allow(dead_code)]
    pub fn z(&self, e1: bool, e2: bool, e3: bool, e4: bool, e5: bool, e6: bool) -> bool {
        let mut result = e1 ^ e5 ^ e6 ^ (e1 & e4) ^ (e2 & e3) ^ (e2 & e5) ^ (e4 & e5) ^ (e5 & e6);
        result ^= (e1 & e3 & e4) ^ (e1 & e3 & e6) ^ (e1 & e4 & e5) ^ (e2 & e3 & e6) ^ (e2 & e4 & e6) ^ (e3 & e5 & e6);
        result ^= (e1 & e2 & e3 & e4) ^(e1 & e2 & e3 & e5) ^ (e1 & e2 & e5 & e6) ^ (e2 & e3 & e4 & e6) ^(e1 & e2 & e3 & e4 & e5);
        result ^= (e1 & e3 & e4 & e5 & e6);

        result
    }
    #[allow(dead_code)]
    pub fn get_f_bit_and_rotate(&mut self) -> bool {
        let new_bit = self.f_vector[0] ^ self.f_vector[1] ^ self.f_vector[2] ^ self.f_vector[5];
        self.f_vector.rotate_right(1);
        self.f_vector[0] = new_bit;
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

    pub fn encrypt_character_simple(&mut self, char: [bool; 5]) -> [bool; 5] {
        self.single_round();
        let srv_2: [bool; 5] = [self.a[0], self.a[1], self.a[2], self.a[3], self.a[4]];

        [
            srv_2[0] ^ char[0],
            srv_2[1] ^ char[1],
            srv_2[2] ^ char[2],
            srv_2[3] ^ char[3],
            srv_2[4] ^ char[4],
        ]
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
    pub fn single_round(&mut self) {
        let mut t_array = [false; 10];
        for outer_round in 0..12 {
            for inner_round in 0..126 {
                //    for inner_round in 0..2 {
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
                        self.get_u(self.p[3 - 2] as usize - 1),
                        self.get_u(self.p[4 - 2] as usize - 1),
                        self.get_u(self.p[5 - 2] as usize - 1),
                        self.get_u(self.p[6 - 2] as usize - 1),
                    );
                t_array[1] = self.get_f_bit_and_rotate();

                // Shift U vector right by one and insert 9 bits from T into U
                self.u_vector.rotate_right(1);
                let old_u = self.u_vector.clone();
                for j in 1..10 {
                    self.u_vector[4 * j - 3] = old_u[self.d[j - 1] as usize] ^ t_array[10 - j];
                }
                let s1_bit = self.get_s1_bit();
                self.u_vector[0] = s1_bit;
                // Shift s1 and s2
                self.s1.rotate_right(1);
                self.s2.rotate_right(1);
            }
            self.a[outer_round] = self.u_vector[self.alpha as usize - 1]
        }
    }
}
