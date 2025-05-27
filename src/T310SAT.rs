use rustsat::{
    instances::SatInstance,
    types::{Clause, Lit},
};

pub struct T310SAT {
    // Key components: two 120-bit sequences
    s1: [bool; 120],
    s2: [bool; 120],

    // Standard U-Vector: 37-bit register
    u_vector: [bool; 37],
    // 61-bit synchronization sequence (initialization vector (or F - Vector LFSR))
    f_vector: [bool; 61],

    //Long Term Key
    // P function mapping {1,2,...27} to {1,...36}
    p: [u8; 27],
    // D function mapping {1,2,...9} to {0,1,...36}
    d: [u8; 9],
    alpha: u8,
    a: [bool; 13],
    sat_instance: SatInstance,
}

impl T310SAT {
    pub fn new(s1: &[bool; 120], s2: &[bool; 120], iv: &[bool; 61]) -> Self {
        // Validate the key components

        //assert!(iv != 0, "IV must not be all zeros");

        // Convert IV to BitVec

        // Initialize with the standard U-Vector
        let standard_u_vector: [bool; 37] = [
            false, true, true, false, true, false, false, true, true, true, false, false, false,
            true, true, true, true, true, false, false, true, false, false, false, false, true,
            false, true, true, false, true, false, false, false, true, true, false,
        ];

        // Longterm Key LZS-31 approved ()
        let p: [u8; 27] = [
            7, 4, 33, 30, 18, 36, 5, 35, 9, 16, 23, 26, 32, 12, 21, 1, 13, 25, 20, 8, 24, 15, 22,
            29, 10, 28, 6,
        ];
        let d: [u8; 9] = [0, 15, 3, 23, 11, 27, 31, 35, 19];
        let alpha: u8 = 2; // {1,2,3,...36}
        let a = [false; 13];
        /*
        println!("s1: {:?}", s1);
        println!("s2: {:?}", s2);
        println!("standard_u_vector: {:?}", standard_u_vector);
        println!("iv_bitvec: {:?}", iv);
        */
        let sat_instance = SatInstance::new();
        Self {
            s1: s1.clone(),
            s2: s2.clone(),
            u_vector: standard_u_vector,
            f_vector: iv.clone(),
            p,
            d,
            alpha,
            a,
            sat_instance,
        }
    }

    #[allow(dead_code)]
    fn z(&mut self, e1: Lit, e2: Lit, e3: Lit, e4: Lit, e5: Lit, e6: Lit) -> Lit {
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
            or_clauses.push(or_output);
        }
        self.sat_instance.add_cube_impl_lit(&or_clauses, output);
        output
    }
    /*
    #[allow(dead_code)]
    fn get_f_bit_and_rotate(&mut self) -> bool {
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
    fn single_round(&mut self) {
        let mut t_array = [false; 10];
        for outer_round in 0..12 {
            for inner_round in 0..126 {
                //for inner_round in 0..2 {
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
    */
}

#[cfg(test)]
mod tests {
    use crate::T310::T310Cipher;

    use super::*;
    use rustsat::instances::ManageVars;
    use rustsat::solvers::{Solve, SolverResult};
    use rustsat::types::{Lit, TernaryVal, Var};

    #[test]
    fn brute_force_test_z_function() {
        for i in 0..64 {
            let mut t310_sat = T310SAT::new(&[false; 120], &[false; 120], &[false; 61]);
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
}
