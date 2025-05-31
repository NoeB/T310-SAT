use rustsat::{
    encodings::card::{BoundLower, DefLowerBounding},
    instances::{Cnf, SatInstance},
    types::{Clause, Lit, Var},
};

pub struct T310SAT {
    // Key components: two 120-bit sequences
    s1: [Lit; 120],
    s2: [Lit; 120],
    initial_s1: [Lit; 120],
    initial_s2: [Lit; 120],

    // Standard U-Vector: 36-bit register
    u_vector: [Lit; 36],
    // 61-bit synchronization sequence (initialization vector (or F - Vector LFSR))
    f_vector: [Lit; 61],
    initial_iv: [Lit; 61],

    //Long Term Key
    // P function mapping {1,2,...27} to {1,...36}
    p: [u8; 27],
    // D function mapping {1,2,...9} to {0,1,...36}
    d: [u8; 9],
    alpha: u8,
    a: [Lit; 13],
    sat_instance: SatInstance,
}

impl T310SAT {
    pub fn new(
        s1: Option<&[bool; 120]>,
        s2: Option<&[bool; 120]>,
        iv: Option<&[bool; 61]>,
    ) -> Self {
        let standard_u_vector: [bool; 36] = [
            false, true, true, false, true, false, false, true, true, true, false, false, false,
            true, true, true, true, true, false, false, true, false, false, false, false, true,
            false, true, true, false, true, false, false, false, true, true,
        ];

        let p: [u8; 27] = [
            7, 4, 33, 30, 18, 36, 5, 35, 9, 16, 23, 26, 32, 12, 21, 1, 13, 25, 20, 8, 24, 15, 22,
            29, 10, 28, 6,
        ];
        let d: [u8; 9] = [0, 15, 3, 23, 11, 27, 31, 35, 19];
        let alpha: u8 = 2;
        let a = [false; 13];

        let mut sat_instance = SatInstance::new();
        let standard_u_vector_lits = convert_to_lits(&standard_u_vector, &mut sat_instance);
        let a_lits = convert_to_lits(&a, &mut sat_instance);

        // Handle IV
        let (iv_lits, initial_iv) = match iv {
            Some(iv_values) => {
                let lits = convert_to_lits(iv_values, &mut sat_instance);
                (
                    lits.clone().try_into().expect("IV must have 61 elements"),
                    lits.try_into().expect("IV must have 61 elements"),
                )
            }
            None => {
                let lits: [Lit; 61] = std::array::from_fn(|_| sat_instance.new_lit());
                sat_instance.add_nary(&lits.to_vec()); // At least 1 must be 1
                (lits, lits)
            }
        };

        // Handle S1
        let (s1_lits, initial_s1) = match s1 {
            Some(s1_values) => {
                let lits = convert_to_lits(s1_values, &mut sat_instance);
                let arr: [Lit; 120] = lits.try_into().expect("s1 must have 120 elements");
                (arr, arr.clone())
            }
            None => {
                let lits: [Lit; 120] = std::array::from_fn(|_| sat_instance.new_lit());
                (lits, lits.clone())
            }
        };

        // Handle S2
        let (s2_lits, initial_s2) = match s2 {
            Some(s2_values) => {
                let lits = convert_to_lits(s2_values, &mut sat_instance);
                let arr: [Lit; 120] = lits.try_into().expect("s2 must have 120 elements");
                (arr, arr.clone())
            }
            None => {
                let lits: [Lit; 120] = std::array::from_fn(|_| sat_instance.new_lit());
                (lits, lits.clone())
            }
        };

        Self {
            initial_s1,
            s1: s1_lits,
            initial_s2,
            s2: s2_lits,
            u_vector: standard_u_vector_lits
                .try_into()
                .expect("standard_u_vector_lits must have 36 elements"),
            f_vector: iv_lits.clone(),
            initial_iv,
            p,
            d,
            alpha,
            a: a_lits.try_into().expect("a lits must contain 13 elements"),
            sat_instance,
        }
    }

    #[allow(dead_code)]
    fn z_2(&mut self, e1: Lit, e2: Lit, e3: Lit, e4: Lit, e5: Lit, e6: Lit) -> Lit {
        let output = self.sat_instance.new_lit();

        let clauses = vec![
            vec![e1, e2, !e3, e4, !e5, !e6],
            vec![e1, !e2, e3, !e4, !e5, !e6],
            vec![e1, !e2, e3, !e4, e5, !e6],
            vec![!e1, !e2, e3, !e4, e5, !e6],
            vec![!e1, !e2, e3, !e4, !e5, e6],
            vec![e1, e2, e3, !e4, !e5, !e6],
            vec![e1, e2, e5, e6],
            vec![e1, e2, !e4, !e5, e6],
            vec![e1, e3, e5, e6],
            vec![e1, !e2, e3, e4, !e5, e6],
            vec![e1, !e2, !e3, !e4, !e5, e6],
            vec![!e1, e3, e4, e5, !e6],
            vec![e1, !e2, e3, e4, !e5, !e6],
            vec![!e1, e2, e3, e4, !e5, !e6],
            vec![!e1, e3, !e4, e5, e6],
            vec![!e1, !e2, !e3, e4, e5, e6],
            vec![!e1, e2, e4, !e5, e6],
            vec![!e1, e2, !e3, !e4, !e5, e6],
            vec![!e1, !e2, !e3, !e4, e5, !e6],
            vec![!e1, e2, !e3, e4, !e5, !e6],
            vec![e1, !e2, !e3, !e4, !e5, !e6],
            vec![!e1, !e2, e3, e4, !e5, !e6],
            vec![!e1, !e2, e3, !e4, !e5, !e6],
            vec![!e1, !e2, !e3, !e4, !e5, !e6],
        ];
        let mut or_clauses = vec![];
        for clause in &clauses {
            let or_output = self.sat_instance.new_lit();
            self.sat_instance.add_clause_impl_lit(&clause, or_output);
            self.sat_instance.add_lit_impl_clause(or_output, clause);
            or_clauses.push(or_output);
        }
        self.sat_instance.add_cube_impl_lit(&or_clauses, output);
        self.sat_instance.add_lit_impl_cube(output, &or_clauses);
        output
    }

    #[allow(dead_code)]
    fn z(&mut self, e1: Lit, e2: Lit, e3: Lit, e4: Lit, e5: Lit, e6: Lit) -> Lit {
        let output = self.sat_instance.new_lit();
        let clauses = vec![
            vec![e3, !e4, !e5, e6],
            vec![e1, e2, e3, e4, !e6],
            vec![e2, e3, !e4, e5],
            vec![!e1, !e4, !e5, e6],
            vec![!e2, e4, !e5, e6],
            vec![!e1, e2, !e3, e4, e5, !e6],
            vec![e1, !e2, e3, !e5],
            vec![!e1, !e2, !e4, e5, !e6],
            vec![!e2, e3, e4, e6],
            vec![!e1, e2, e3, !e5],
            vec![e1, !e2, !e3, e4, e5],
            vec![e1, e2, !e4, e5, !e6],
            vec![!e1, !e2, !e3, !e4, e6],
            vec![e1, !e3, !e4, !e5, !e6],
        ];
        let mut and_clauses = vec![];
        for clause in &clauses {
            let and_output = self.sat_instance.new_lit();
            self.sat_instance.add_cube_impl_lit(&clause, and_output);
            self.sat_instance.add_lit_impl_cube(and_output, clause);
            and_clauses.push(and_output);
        }
        self.sat_instance.add_clause_impl_lit(&and_clauses, output);
        self.sat_instance.add_lit_impl_clause(output, &and_clauses);
        output
    }

    fn get_f_bit_and_rotate(&mut self) -> Lit {
        let f0 = self.f_vector[0];
        let f1 = self.f_vector[1];
        let f2 = self.f_vector[2];
        let f5 = self.f_vector[5];

        // Chain XORs: ((f0 ⊕ f1) ⊕ f2) ⊕ f5
        let xor_01 = self.xor2(f0, f1);
        let xor_012 = self.xor2(xor_01, f2);
        let output = self.xor2(xor_012, f5);

        self.f_vector.rotate_right(1);
        self.f_vector[0] = output;
        output
    }
    #[allow(dead_code)]
    fn get_f_bit_and_rotate_2(&mut self) -> Lit {
        let output = self.sat_instance.new_lit();
        let f0 = self.f_vector[0];
        let f1 = self.f_vector[1];
        let f2 = self.f_vector[2];
        let f5 = self.f_vector[5];

        /*
        And(
            Or(f0, f1, !f2, !f5),
            Or(!f0, !f1, !f2, !f5),
            Or(f0, !f1, f2, !f5),
            Or(!f0, f1, f2, !f5),
            Or(f0, !f1, !f2, f5),
            Or(!f0, f1, !f2, f5),
            Or(f0, f1, f2, f5),
            Or(!f0, !f1, f2, f5))
        */
        let clauses = vec![
            vec![f0, f1, !f2, !f5],
            vec![!f0, !f1, !f2, !f5],
            vec![f0, !f1, f2, !f5],
            vec![!f0, f1, f2, !f5],
            vec![f0, !f1, !f2, f5],
            vec![!f0, f1, !f2, f5],
            vec![f0, f1, f2, f5],
            vec![!f0, !f1, f2, f5],
        ];
        let mut or_clauses = vec![];
        for clause in &clauses {
            let or_output = self.sat_instance.new_lit();
            self.sat_instance.add_clause_impl_lit(&clause, or_output);
            self.sat_instance.add_lit_impl_clause(or_output, clause);
            or_clauses.push(or_output);
        }
        self.sat_instance.add_cube_impl_lit(&or_clauses, output);
        self.sat_instance.add_lit_impl_cube(output, &or_clauses);
        self.f_vector.rotate_right(1);
        self.f_vector[0] = output;
        output
    }

    //get the last bit of the s2 vector
    #[allow(dead_code)]
    fn get_s2_bit(&self) -> Lit {
        self.s2[119]
    }
    // get the last bit of the s2 vector
    #[allow(dead_code)]
    fn get_s1_bit(&self) -> Lit {
        self.s1[119]
    }
    /*

            fn shift_srv(srv: &mut [bool; 5]) {
                let feedback_bit = srv[0] ^ srv[2];
                srv.rotate_left(1);
                srv[4] = feedback_bit;
            }
    */

    pub fn encrypt_character_simple_lit(&mut self, char_lit: [Lit; 5]) -> [Lit; 5] {
        self.single_round();
        let srv_2: [Lit; 5] = [self.a[0], self.a[1], self.a[2], self.a[3], self.a[4]];

        [
            self.xor2(srv_2[0], char_lit[0]),
            self.xor2(srv_2[1], char_lit[1]),
            self.xor2(srv_2[2], char_lit[2]),
            self.xor2(srv_2[3], char_lit[3]),
            self.xor2(srv_2[4], char_lit[4]),
        ]
    }
    pub fn encrypt_character_simple(&mut self, char: [bool; 5]) -> [Lit; 5] {
        let char_lit = convert_to_lits(&char, &mut self.sat_instance);
        self.encrypt_character_simple_lit(
            char_lit.try_into().expect("Must be 5 exactly 5 elements"),
        )
    }
    /*
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
    */
    fn get_u(&self, p: usize) -> Lit {
        if p == 0 {
            return self.get_s1_bit();
        }
        self.u_vector[p]
    }
    fn xor2(&mut self, a: Lit, b: Lit) -> Lit {
        let output = self.sat_instance.new_lit();

        // XOR(a,b) = output is equivalent to these 4 clauses:
        // (a ∨ b ∨ ¬output)
        // (¬a ∨ ¬b ∨ ¬output)
        // (¬a ∨ b ∨ output)
        // (a ∨ ¬b ∨ output)

        // When output is true: (a ∨ b) ∧ (¬a ∨ ¬b)
        self.sat_instance.add_lit_impl_clause(output, &vec![a, b]);
        self.sat_instance.add_lit_impl_clause(output, &vec![!a, !b]);

        // When output is false: (¬a ∨ ¬b) ∧ (a ∨ b)
        // Which is equivalent to: (a ∨ ¬b) ∧ (¬a ∨ b)
        self.sat_instance.add_lit_impl_clause(!output, &vec![!a, b]);
        self.sat_instance.add_lit_impl_clause(!output, &vec![a, !b]);

        output
    }
    fn xor22(&mut self, a: Lit, b: Lit) -> Lit {
        let output = self.sat_instance.new_lit();
        //And(Or(t8, u28), Or(~t8, ~u28))

        let clauses = vec![vec![a, b], vec![!a, !b]];
        let mut or_clauses = vec![];
        for clause in &clauses {
            let or_output = self.sat_instance.new_lit();
            self.sat_instance.add_clause_impl_lit(&clause, or_output);
            self.sat_instance.add_lit_impl_clause(or_output, clause);
            or_clauses.push(or_output);
        }
        self.sat_instance.add_cube_impl_lit(&or_clauses, output);
        self.sat_instance.add_lit_impl_cube(output, &or_clauses);
        output
    }

    #[allow(dead_code)]
    fn single_round(&mut self) {
        let mut t_array: [Lit; 10] = convert_to_lits(&[false; 10], &mut self.sat_instance)
            .try_into()
            .expect("t array must have 10 elements");
        for outer_round in 0..12 {
            for inner_round in 0..126 {
                //for inner_round in 0..2 {
                t_array[9] = self.xor2(t_array[8], self.get_u(self.p[28 - 2] as usize - 1));
                let z_9 = self.z(
                    self.get_u(self.p[22 - 2] as usize - 1),
                    self.get_u(self.p[23 - 2] as usize - 1),
                    self.get_u(self.p[24 - 2] as usize - 1),
                    self.get_u(self.p[25 - 2] as usize - 1),
                    self.get_u(self.p[26 - 2] as usize - 1),
                    self.get_u(self.p[27 - 2] as usize - 1),
                );
                t_array[8] = self.xor2(t_array[7], z_9);
                t_array[7] = self.xor2(t_array[6], self.get_u(self.p[21 - 2] as usize - 1));
                let z_6 = self.z(
                    self.get_u(self.p[15 - 2] as usize - 1),
                    self.get_u(self.p[16 - 2] as usize - 1),
                    self.get_u(self.p[17 - 2] as usize - 1),
                    self.get_u(self.p[18 - 2] as usize - 1),
                    self.get_u(self.p[19 - 2] as usize - 1),
                    self.get_u(self.p[20 - 2] as usize - 1),
                );
                t_array[6] = self.xor2(t_array[5], z_6);
                t_array[5] = self.xor2(t_array[4], self.get_u(self.p[14 - 2] as usize - 1));
                let z_4 = self.z(
                    self.get_u(self.p[8 - 2] as usize - 1),
                    self.get_u(self.p[9 - 2] as usize - 1),
                    self.get_u(self.p[10 - 2] as usize - 1),
                    self.get_u(self.p[11 - 2] as usize - 1),
                    self.get_u(self.p[12 - 2] as usize - 1),
                    self.get_u(self.p[13 - 2] as usize - 1),
                );
                t_array[4] = self.xor2(t_array[3], z_4);
                t_array[3] = self.xor2(t_array[2], self.get_u(self.p[7 - 2] as usize - 1));
                let z_2 = self.z(
                    self.get_s2_bit(),
                    self.get_u(self.p[2 - 2] as usize - 1),
                    self.get_u(self.p[3 - 2] as usize - 1),
                    self.get_u(self.p[4 - 2] as usize - 1),
                    self.get_u(self.p[5 - 2] as usize - 1),
                    self.get_u(self.p[6 - 2] as usize - 1),
                );
                t_array[2] = self.xor2(t_array[1], z_2);
                t_array[1] = self.get_f_bit_and_rotate();

                // Shift U vector right by one and insert 9 bits from T into U
                self.u_vector.rotate_right(1);
                let old_u = self.u_vector.clone();
                for j in 1..10 {
                    self.u_vector[4 * j - 3] =
                        self.xor2(old_u[self.d[j - 1] as usize], t_array[10 - j]);
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

fn convert_to_lits(iv: &[bool], sat_instance: &mut SatInstance) -> Vec<Lit> {
    let iv_vars: Vec<Var> = iv.iter().map(|_| sat_instance.new_var()).collect();
    let iv_lits: Vec<Lit> = iv_vars.iter().map(|&v| v.pos_lit()).collect();

    for (bit, lit) in iv.iter().zip(iv_lits.iter()) {
        let assigned_lit = if *bit { *lit } else { !*lit };
        sat_instance.add_unit(assigned_lit);
    }
    iv_lits
}

#[cfg(test)]
mod tests {

    use std::cmp;

    use crate::T310::T310Cipher;

    use super::*;
    use crate::CCITT2::SimpleCCITT2;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rustsat::instances::ManageVars;
    use rustsat::solvers::{self, Solve, SolveStats, SolverResult};
    use rustsat::types::{Lit, TernaryVal, Var};

    #[test]
    fn brute_force_test_z_function() {
        for i in 0..64 {
            let mut t310_sat =
                T310SAT::new(Some(&[false; 120]), Some(&[false; 120]), Some(&[false; 61]));
            let t310 = T310Cipher::new(&[false; 120], &[false; 120], &[false; 61]);

            let input_bits: [bool; 6] = [
                (i & 0b000001) != 0,
                (i & 0b000010) != 0,
                (i & 0b000100) != 0,
                (i & 0b001000) != 0,
                (i & 0b010000) != 0,
                (i & 0b100000) != 0,
            ];

            // Create SAT vars for e1..e6
            let vars: Vec<Var> = (0..6).map(|_| t310_sat.sat_instance.new_var()).collect();
            let lits: Vec<Lit> = vars.iter().map(|&v| v.pos_lit()).collect();

            // Force values of inputs
            for (bit, lit) in input_bits.iter().zip(lits.iter()) {
                let assigned_lit = if *bit { *lit } else { !*lit };
                t310_sat.sat_instance.add_unit(assigned_lit);
            }

            // Call z function with fixed literals
            let z_out = t310_sat.z(lits[0], lits[1], lits[2], lits[3], lits[4], lits[5]);

            // Now solve the CNF to get value of z_out
            let mut solver = rustsat_cadical::CaDiCaL::default();
            let (cnf, vm) = t310_sat.sat_instance.into_cnf();
            if let Some(max_var) = vm.max_var() {
                solver.reserve(max_var).unwrap();
            }
            solver.add_cnf(cnf).unwrap();

            let result = solver.solve().expect("SAT expected");
            assert_eq!(result, SolverResult::Sat);
            let sol = solver.full_solution().unwrap();
            println!("output sat: {}", sol[z_out.var()]);
            println!(
                "real z output: {:?}",
                t310.z(
                    input_bits[0],
                    input_bits[1],
                    input_bits[2],
                    input_bits[3],
                    input_bits[4],
                    input_bits[5]
                )
            );
            let expected: TernaryVal = match t310.z(
                input_bits[0],
                input_bits[1],
                input_bits[2],
                input_bits[3],
                input_bits[4],
                input_bits[5],
            ) {
                true => TernaryVal::True,
                false => TernaryVal::False,
            };
            assert_eq!(sol[z_out.var()], expected);

            println!("---------------")
        }
    }

    fn random_bool_array_61(seed: u64) -> [bool; 61] {
        let mut arr = [false; 61];
        let mut rng = StdRng::seed_from_u64(seed);
        for i in 0..61 {
            arr[i] = rng.random();
        }
        arr
    }
    fn random_bool_array_120(seed: u64) -> [bool; 120] {
        let mut arr = [false; 120];
        let mut rng = StdRng::seed_from_u64(seed);
        for i in 0..120 {
            arr[i] = rng.random();
        }
        arr
    }

    #[test]
    fn brute_force_test_f_bit_function() {
        for i in 0..5000 {
            let iv = random_bool_array_61(i);
            let mut t310_sat = T310SAT::new(Some(&[false; 120]), Some(&[false; 120]), Some(&iv));
            let mut t310 = T310Cipher::new(&[false; 120], &[false; 120], &iv);
            //println!("------");
            for _round in 0..100 {
                let f_bit_out = t310_sat.get_f_bit_and_rotate();
                let mut solver = rustsat_cadical::CaDiCaL::default();
                let (cnf, vm) = t310_sat.sat_instance.clone().into_cnf();
                if let Some(max_var) = vm.max_var() {
                    solver.reserve(max_var).unwrap();
                }
                solver.add_cnf(cnf).unwrap();

                let result = solver.solve().expect("SAT expected");
                assert_eq!(result, SolverResult::Sat);
                let sol = solver.full_solution().unwrap();
                //println!("output sat: {}", sol[f_bit_out.var()]);
                let expected: TernaryVal = match t310.get_f_bit_and_rotate() {
                    true => TernaryVal::True,
                    false => TernaryVal::False,
                };

                //println!("expected output {}", expected);
                assert_eq!(sol[f_bit_out.var()], expected);
            }
        }
    }

    #[test]
    fn brute_force_test_single_round_function() {
        let start = std::time::Instant::now();
        for i in 0..500 {
            let iv = random_bool_array_61(i);
            let mut t310_sat = T310SAT::new(Some(&[false; 120]), Some(&[false; 120]), Some(&iv));
            let mut t310 = T310Cipher::new(&[false; 120], &[false; 120], &iv);
            t310_sat.single_round();
            t310.single_round();
            let mut solver = rustsat_cadical::CaDiCaL::default();
            //let mut solver = rustsat_kissat::Kissat::default();
            //let mut solver = rustsat_minisat::core::Minisat::default();
            //let mut solver = rustsat_batsat::BasicSolver::default();
            //let mut solver = rustsat_glucose::core::Glucose::default();
            //let mut solver = rustsat_cryptominisat::CryptoMiniSat::default();
            let (cnf, vm) = t310_sat.sat_instance.clone().into_cnf();
            if let Some(max_var) = vm.max_var() {
                println!("MaxVar: {}", max_var);
                solver.reserve(max_var).unwrap();
            }
            solver.add_cnf(cnf).unwrap();

            let result = solver.solve().expect("SAT expected");
            assert_eq!(result, SolverResult::Sat);
            let sol = solver.full_solution().unwrap();
            let expected_a: Vec<TernaryVal> = t310
                .get_a()
                .iter()
                .map(|&b| match b {
                    true => TernaryVal::True,
                    false => TernaryVal::False,
                })
                .collect();
            let sat_output: Vec<TernaryVal> = t310_sat.a.iter().map(|i| sol[i.var()]).collect();
            //println!("expected a: {:?}", expected_a);
            //println!("sat a {:?}", sat_output);
            assert_eq!(sat_output, expected_a);
        }
        let duration = start.elapsed();
        println!("test_encrypt_simple took {:?}", duration);
    }
    #[test]
    fn test_xor2() {
        let cases = [(false, false), (false, true), (true, false), (true, true)];

        for &(a_val, b_val) in &cases {
            let expected_bool = a_val ^ b_val;
            let expected_bool2 = expected_bool ^ b_val;
            let mut t310_sat =
                T310SAT::new(Some(&[false; 120]), Some(&[false; 120]), Some(&[false; 61]));
            let a_var = t310_sat.sat_instance.new_var();
            let b_var = t310_sat.sat_instance.new_var();
            let a_lit = a_var.pos_lit();
            let b_lit = b_var.pos_lit();

            // Assign values to a and b
            t310_sat
                .sat_instance
                .add_unit(if a_val { a_lit } else { !a_lit });
            t310_sat
                .sat_instance
                .add_unit(if b_val { b_lit } else { !b_lit });

            let out: Lit = t310_sat.xor2(a_lit, b_lit);
            let out2 = t310_sat.xor2(out, b_lit);

            let mut solver = rustsat_cadical::CaDiCaL::default();
            let (cnf, vm) = t310_sat.sat_instance.into_cnf();
            if let Some(max_var) = vm.max_var() {
                solver.reserve(max_var).unwrap();
            }
            solver.add_cnf(cnf).unwrap();

            let result = solver.solve().expect("SAT expected");
            assert_eq!(result, rustsat::solvers::SolverResult::Sat);
            let sol = solver.full_solution().unwrap();
            let out_val = sol[out.var()];
            let out_val2 = sol[out2.var()];
            let expected = if expected_bool {
                rustsat::types::TernaryVal::True
            } else {
                rustsat::types::TernaryVal::False
            };
            let expected2 = if expected_bool2 {
                rustsat::types::TernaryVal::True
            } else {
                rustsat::types::TernaryVal::False
            };
            assert_eq!(out_val, expected, "xor2({a_val}, {b_val}) failed");
            assert_eq!(out_val2, expected2, "xor2({expected_bool}, {b_val}) failed");
        }
    }
    fn convert_bool_to_tenary(bool: bool) -> TernaryVal {
        if bool {
            rustsat::types::TernaryVal::True
        } else {
            rustsat::types::TernaryVal::False
        }
    }
    #[test]
    fn test_encrypt_simple() {
        for i in 0..10 {
            let codec = SimpleCCITT2::new();
            let iv = random_bool_array_61(i);
            let s1 = random_bool_array_120(i);
            let mut t310_sat = T310SAT::new(Some(&s1), Some(&[false; 120]), Some(&iv));
            let mut t310 = T310Cipher::new(&s1, &[false; 120], &iv);
            let text = "HELLO";
            let bool_array = codec
                .encode_to_bools(text)
                .expect("Input text should be valid");

            let mut encrypted: Vec<bool> = vec![];
            let mut encrypted_lits: Vec<Lit> = vec![];
            for chunk in bool_array.chunks(5) {
                let chunk_array: [bool; 5] = chunk.try_into().expect("Chunk size must be 5");
                encrypted.extend_from_slice(&t310.encrypt_character_simple(chunk_array));
                encrypted_lits.extend_from_slice(&t310_sat.encrypt_character_simple(chunk_array));
            }
            let mut solver = rustsat_cadical::CaDiCaL::default();
            let (cnf, vm) = t310_sat.sat_instance.into_cnf();
            if let Some(max_var) = vm.max_var() {
                solver.reserve(max_var).unwrap();
            }
            solver.add_cnf(cnf).unwrap();

            let result = solver.solve().expect("SAT expected");
            assert_eq!(result, rustsat::solvers::SolverResult::Sat);
            let sol = solver.full_solution().unwrap();
            let solver_output: Vec<TernaryVal> =
                encrypted_lits.iter().map(|lit| sol[lit.var()]).collect();
            let encryped_ternary: Vec<TernaryVal> = encrypted
                .iter()
                .map(|b| convert_bool_to_tenary(*b))
                .collect();

            //println!("encryped_ternary: {:?}", encryped_ternary);
            //println!("encryped_ternary: {:?}", solver_output);
            assert_eq!(encryped_ternary, solver_output);
        }
    }
    #[test]
    fn test_encrypt_simple_reverse() {
        for i in 0..10 {
            let codec = SimpleCCITT2::new();
            let iv = random_bool_array_61(i);
            let s1 = random_bool_array_120(i);
            let mut t310_sat = T310SAT::new(Some(&s1), Some(&[false; 120]), Some(&iv));
            let mut t310 = T310Cipher::new(&s1, &[false; 120], &iv);
            let text = "HELLO";
            let bool_array = codec
                .encode_to_bools(text)
                .expect("Input text should be valid");

            let mut encrypted: Vec<bool> = vec![];
            let mut encrypted_lits: Vec<Lit> = vec![];
            let mut bool_array_lits: Vec<Lit> = vec![];
            for chunk in bool_array.chunks(5) {
                let chunk_array: [bool; 5] = chunk.try_into().expect("Chunk size must be 5");
                let encrypted_chunk = t310.encrypt_character_simple(chunk_array);
                let input_chunk_lit: [Lit; 5] =
                    std::array::from_fn(|_| t310_sat.sat_instance.new_lit());
                bool_array_lits.extend_from_slice(&input_chunk_lit);
                encrypted.extend_from_slice(&encrypted_chunk);
                let encrypted_chunk_lits: [Lit; 5] =
                    t310_sat.encrypt_character_simple_lit(input_chunk_lit);
                encrypted_lits.extend_from_slice(&encrypted_chunk_lits);

                for (bit, lit) in encrypted_chunk.iter().zip(encrypted_chunk_lits.iter()) {
                    let assigned_lit = if *bit { *lit } else { !*lit };
                    t310_sat.sat_instance.add_unit(assigned_lit);
                }
            }
            let mut solver = rustsat_cadical::CaDiCaL::default();

            let (cnf, vm) = t310_sat.sat_instance.into_cnf();
            if let Some(max_var) = vm.max_var() {
                solver.reserve(max_var).unwrap();
            }
            solver.add_cnf(cnf).unwrap();

            let result = solver.solve().expect("SAT expected");
            assert_eq!(result, rustsat::solvers::SolverResult::Sat);
            let sol = solver.full_solution().unwrap();
            let solver_output: Vec<TernaryVal> =
                encrypted_lits.iter().map(|lit| sol[lit.var()]).collect();
            let encryped_ternary: Vec<TernaryVal> = encrypted
                .iter()
                .map(|b| convert_bool_to_tenary(*b))
                .collect();

            //println!("encryped_ternary: {:?}", encryped_ternary);
            //println!("encryped_ternary: {:?}", solver_output);
            assert_eq!(encryped_ternary, solver_output);

            let input_ternary: Vec<TernaryVal> = bool_array
                .iter()
                .map(|b| convert_bool_to_tenary(*b))
                .collect();
            let solver_input_bool_array_lits: Vec<TernaryVal> =
                bool_array_lits.iter().map(|lit| sol[lit.var()]).collect();
            assert_eq!(input_ternary, solver_input_bool_array_lits);
        }
    }

    #[test]
    fn test_recover_iv() {
        for i in 0..1 {
            let codec = SimpleCCITT2::new();
            let iv: [bool; 61] = random_bool_array_61(i);
            let s1 = random_bool_array_120(i);
            let s2 = random_bool_array_120(i + 1000);
            let mut t310_sat = T310SAT::new(Some(&s1), Some(&s2), None);
            let mut t310 = T310Cipher::new(&s1, &s2, &iv);
            let text = "HELLO Test Encryption";
            let bool_array = codec
                .encode_to_bools(text)
                .expect("Input text should be valid");

            let mut encrypted: Vec<bool> = vec![];
            let mut encrypted_lits: Vec<Lit> = vec![];
            let mut bool_array_lits: Vec<Lit> = vec![];
            for chunk in bool_array.chunks(5) {
                let chunk_array: [bool; 5] = chunk.try_into().expect("Chunk size must be 5");
                let encrypted_chunk = t310.encrypt_character_simple(chunk_array);
                encrypted.extend_from_slice(&encrypted_chunk);
                //encrypted_lits.extend_from_slice(&t310_sat.encrypt_character_simple(chunk_array));

                let input_chunk_lit: [Lit; 5] =
                    std::array::from_fn(|_| t310_sat.sat_instance.new_lit());

                let encrypted_chunk_lits: [Lit; 5] =
                    t310_sat.encrypt_character_simple_lit(input_chunk_lit);
                bool_array_lits.extend_from_slice(&input_chunk_lit);

                encrypted_lits.extend_from_slice(&encrypted_chunk_lits);
                //Set encrypted text equal to real ciphertext
                for (bit, lit) in encrypted_chunk.iter().zip(encrypted_chunk_lits.iter()) {
                    let assigned_lit = if *bit { *lit } else { !*lit };
                    t310_sat.sat_instance.add_unit(assigned_lit);
                }
                //Set plaintext
                for (bit, lit) in chunk.iter().zip(input_chunk_lit.iter()) {
                    let assigned_lit = if *bit { *lit } else { !*lit };
                    t310_sat.sat_instance.add_unit(assigned_lit);
                }
            }

            let mut solver = rustsat_cadical::CaDiCaL::default();
            let (cnf, vm) = t310_sat.sat_instance.into_cnf();
            if let Some(max_var) = vm.max_var() {
                solver.reserve(max_var).unwrap();
            }
            solver.add_cnf(cnf).unwrap();

            let result = solver.solve().expect("SAT expected");
            assert_eq!(result, rustsat::solvers::SolverResult::Sat);
            let sol = solver.full_solution().unwrap();
            let solver_output: Vec<TernaryVal> =
                encrypted_lits.iter().map(|lit| sol[lit.var()]).collect();
            let encryped_ternary: Vec<TernaryVal> = encrypted
                .iter()
                .map(|b| convert_bool_to_tenary(*b))
                .collect();
            let solver_output_bool_array_lits: Vec<TernaryVal> =
                bool_array_lits.iter().map(|lit| sol[lit.var()]).collect();
            println!("bool array: {:?}", bool_array);
            println!("bool_array_lits: {:?}", solver_output_bool_array_lits);
            //println!("encryped_ternary: {:?}", encryped_ternary);
            //println!("encryped_ternary: {:?}", solver_output);
            assert_eq!(encryped_ternary, solver_output);
            /*
            // Not sure if it makes sense to compare them completly because I am not sure if I implemented the iv / f vector correctly
            //Would probably make more sense to to only compare the bits which are really used
            let expected_iv_ternary: Vec<TernaryVal> =
                iv.iter().map(|b| convert_bool_to_tenary(*b)).collect();
            let iv_solver_output: Vec<TernaryVal> = t310_sat
                .initial_iv
                .iter()
                .map(|lit| sol[lit.var()])
                .collect();
            assert_eq!(expected_iv_ternary, iv_solver_output);
            */
        }
    }

    #[test]
    fn test_recover_s2() {
        for i in 0..1 {
            let codec = SimpleCCITT2::new();
            let iv: [bool; 61] = random_bool_array_61(i);
            let s1 = random_bool_array_120(i);
            let s2 = random_bool_array_120(i + 1000);
            let mut t310_sat = T310SAT::new(Some(&s1), None, None);
            let mut t310 = T310Cipher::new(&s1, &s2, &iv);
            let text = "HEY";
            let bool_array = codec
                .encode_to_bools(text)
                .expect("Input text should be valid");

            let mut encrypted: Vec<bool> = vec![];
            let mut encrypted_lits: Vec<Lit> = vec![];
            let mut bool_array_lits: Vec<Lit> = vec![];
            for chunk in bool_array.chunks(5) {
                let chunk_array: [bool; 5] = chunk.try_into().expect("Chunk size must be 5");
                let encrypted_chunk = t310.encrypt_character_simple(chunk_array);
                encrypted.extend_from_slice(&encrypted_chunk);
                //encrypted_lits.extend_from_slice(&t310_sat.encrypt_character_simple(chunk_array));

                let input_chunk_lit: [Lit; 5] =
                    std::array::from_fn(|_| t310_sat.sat_instance.new_lit());

                let encrypted_chunk_lits: [Lit; 5] =
                    t310_sat.encrypt_character_simple_lit(input_chunk_lit);
                bool_array_lits.extend_from_slice(&input_chunk_lit);

                encrypted_lits.extend_from_slice(&encrypted_chunk_lits);
                //Set encrypted text equal to real ciphertext
                for (bit, lit) in encrypted_chunk.iter().zip(encrypted_chunk_lits.iter()) {
                    let assigned_lit = if *bit { *lit } else { !*lit };
                    t310_sat.sat_instance.add_unit(assigned_lit);
                }
                //Set plaintext
                for (bit, lit) in chunk.iter().zip(input_chunk_lit.iter()) {
                    let assigned_lit = if *bit { *lit } else { !*lit };
                    t310_sat.sat_instance.add_unit(assigned_lit);
                }
            }
            println!("Encrypted bools2: {:?}", &encrypted);
            //let mut solver = rustsat_minisat::core::Minisat::default();
            let mut solver = rustsat_cadical::CaDiCaL::default();
            /*
            let mut solver = rustsat_cryptominisat::CryptoMiniSat::default();
            solver.set_option(rustsat_cryptominisat::Options::NumThreads(4));
            */
            let (cnf, vm) = t310_sat.sat_instance.into_cnf();
            if let Some(max_var) = vm.max_var() {
                solver.reserve(max_var).unwrap();
            }
            solver.add_cnf(cnf).unwrap();

            let result = solver.solve().expect("SAT expected");
            println!("Stast: {:?}", solver.stats());
            assert_eq!(result, rustsat::solvers::SolverResult::Sat);
            let sol = solver.full_solution().unwrap();
            let solver_output: Vec<TernaryVal> =
                encrypted_lits.iter().map(|lit| sol[lit.var()]).collect();
            let encryped_ternary: Vec<TernaryVal> = encrypted
                .iter()
                .map(|b| convert_bool_to_tenary(*b))
                .collect();
            let solver_output_bool_array_lits: Vec<TernaryVal> =
                bool_array_lits.iter().map(|lit| sol[lit.var()]).collect();
            println!("bool array: {:?}", bool_array);
            println!("bool_array_lits: {:?}", solver_output_bool_array_lits);
            println!("encryped_ternary: {:?}", encryped_ternary);
            println!("encryped_ternary: {:?}", solver_output);
            assert_eq!(encryped_ternary, solver_output);
            println!("iv: {:?}", iv);
            println!("s1: {:?}", s1);
            println!("s2: {:?}", s2);
            let expected_s2_ternary: Vec<TernaryVal> =
                s2.iter().map(|b| convert_bool_to_tenary(*b)).collect();
            let s2_solver_output: Vec<TernaryVal> = t310_sat
                .initial_s2
                .iter()
                .map(|lit| sol[lit.var()])
                .collect();

            let iv_solver_output: Vec<TernaryVal> = t310_sat
                .initial_iv
                .iter()
                .map(|lit| sol[lit.var()])
                .collect();

            println!("Iv solver output: {:?}", iv_solver_output);
            assert_eq!(expected_s2_ternary, s2_solver_output);
        }
    }

    #[test]
    fn test_recover_s2_and_s1() {
        for i in 0..1 {
            let codec = SimpleCCITT2::new();
            let iv: [bool; 61] = random_bool_array_61(i);
            let s1 = random_bool_array_120(i);
            let s2 = random_bool_array_120(i + 1000);
            let mut t310_sat = T310SAT::new(None, None, None);
            let mut t310 = T310Cipher::new(&s1, &s2, &iv);
            let text = "HEY";
            let bool_array = codec
                .encode_to_bools(text)
                .expect("Input text should be valid");

            let mut encrypted: Vec<bool> = vec![];
            let mut encrypted_lits: Vec<Lit> = vec![];
            let mut bool_array_lits: Vec<Lit> = vec![];
            for chunk in bool_array.chunks(5) {
                let chunk_array: [bool; 5] = chunk.try_into().expect("Chunk size must be 5");
                let encrypted_chunk = t310.encrypt_character_simple(chunk_array);
                encrypted.extend_from_slice(&encrypted_chunk);
                //encrypted_lits.extend_from_slice(&t310_sat.encrypt_character_simple(chunk_array));

                let input_chunk_lit: [Lit; 5] =
                    std::array::from_fn(|_| t310_sat.sat_instance.new_lit());

                let encrypted_chunk_lits: [Lit; 5] =
                    t310_sat.encrypt_character_simple_lit(input_chunk_lit);
                bool_array_lits.extend_from_slice(&input_chunk_lit);

                encrypted_lits.extend_from_slice(&encrypted_chunk_lits);
                //Set encrypted text equal to real ciphertext
                for (bit, lit) in encrypted_chunk.iter().zip(encrypted_chunk_lits.iter()) {
                    let assigned_lit = if *bit { *lit } else { !*lit };
                    t310_sat.sat_instance.add_unit(assigned_lit);
                }
                //Set plaintext
                for (bit, lit) in chunk.iter().zip(input_chunk_lit.iter()) {
                    let assigned_lit = if *bit { *lit } else { !*lit };
                    t310_sat.sat_instance.add_unit(assigned_lit);
                }
            }
            println!("Encrypted bools2: {:?}", &encrypted);
            let mut solver = rustsat_cadical::CaDiCaL::default();
            /*
                        let mut solver = rustsat_cryptominisat::CryptoMiniSat::default();
                        solver.set_option(rustsat_cryptominisat::Options::NumThreads(4));
            */
            let (cnf, vm) = t310_sat.sat_instance.into_cnf();
            if let Some(max_var) = vm.max_var() {
                solver.reserve(max_var).unwrap();
            }
            solver.add_cnf(cnf).unwrap();

            let result = solver.solve().expect("SAT expected");
            println!("Stast: {:?}", solver.stats());
            assert_eq!(result, rustsat::solvers::SolverResult::Sat);
            let sol = solver.full_solution().unwrap();
            let solver_output: Vec<TernaryVal> =
                encrypted_lits.iter().map(|lit| sol[lit.var()]).collect();
            let encryped_ternary: Vec<TernaryVal> = encrypted
                .iter()
                .map(|b| convert_bool_to_tenary(*b))
                .collect();
            let solver_output_bool_array_lits: Vec<TernaryVal> =
                bool_array_lits.iter().map(|lit| sol[lit.var()]).collect();
            println!("bool array: {:?}", bool_array);
            println!("bool_array_lits: {:?}", solver_output_bool_array_lits);
            println!("encryped_ternary: {:?}", encryped_ternary);
            println!("encryped_ternary: {:?}", solver_output);
            assert_eq!(encryped_ternary, solver_output);
            println!("iv: {:?}", iv);
            println!("s1: {:?}", s1);
            println!("s2: {:?}", s2);
            let expected_s2_ternary: Vec<TernaryVal> =
                s2.iter().map(|b| convert_bool_to_tenary(*b)).collect();
            let s2_solver_output: Vec<TernaryVal> = t310_sat
                .initial_s2
                .iter()
                .map(|lit| sol[lit.var()])
                .collect();

            let iv_solver_output: Vec<TernaryVal> = t310_sat
                .initial_iv
                .iter()
                .map(|lit| sol[lit.var()])
                .collect();

            let expected_s1_ternary: Vec<TernaryVal> =
                s1.iter().map(|b| convert_bool_to_tenary(*b)).collect();
            let s1_solver_output: Vec<TernaryVal> = t310_sat
                .initial_s1
                .iter()
                .map(|lit| sol[lit.var()])
                .collect();

            println!("expected s1: {:?}", expected_s1_ternary);
            println!("solver s1: {:?}", s1_solver_output);

            println!("Iv solver output: {:?}", iv_solver_output);
            assert_eq!(expected_s2_ternary, s2_solver_output);
        }
    }
}
