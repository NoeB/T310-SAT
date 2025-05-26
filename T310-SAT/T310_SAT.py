import SimpleCCITT2
from pysat.solvers import Glucose3, Cadical195, Glucose42
from pysat.formula import CNF
from pysat.formula import *
from typing import List, Optional, Tuple, Dict
import ast

print("hello")
simpleCCITT2_1 = SimpleCCITT2.SimpleCCITT2()
simpleCCITT2_2 = SimpleCCITT2.SimpleCCITT2()
print(simpleCCITT2_2.py_decode_from_bools(simpleCCITT2_1.py_encode_to_bools("test")))


class T310SatSolver:
    def __init__(self):
        self.cnf = CNF()
        self.var_counter = 1
        self.p = [
            7,
            4,
            33,
            30,
            18,
            36,
            5,
            35,
            9,
            16,
            23,
            26,
            32,
            12,
            21,
            1,
            13,
            25,
            20,
            8,
            24,
            15,
            22,
            29,
            10,
            28,
            6,
        ]
        self.d = [0, 15, 3, 23, 11, 27, 31, 35, 19]
        self.standard_u_vector = [
            False,
            True,
            True,
            False,
            True,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            True,
            True,
            False,
            True,
            False,
            False,
            False,
            True,
            True,
            False,
        ]
        self.alpha = 2
        self.a = [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
        self.variables = {}
        self.excluded_solutions = []

    def new_var(self, name: str = None) -> int:
        var = self.var_counter
        self.var_counter += 1
        if name:
            self.variables[name] = var
        return var

    def add_xor2(self, a: int, b: int, result: int):
        """Add XOR constraint for 2 variables"""
        self.cnf.append([-a, -b, -result])
        self.cnf.append([-a, b, result])
        self.cnf.append([a, -b, result])
        self.cnf.append([a, b, -result])

    def add_and(self, a: int, b: int, result: int):
        """Add AND constraint"""
        self.cnf.append([-result, a])
        self.cnf.append([-result, b])
        self.cnf.append([result, -a, -b])

    def add_and_multi(self, inputs: list, result: int):
        """Add multi-input AND constraint"""
        # result implies all inputs are True
        for inp in inputs:
            self.cnf.append([-result, inp])

        # if any input is False, result is False
        clause = [result]
        for inp in inputs:
            clause.append(-inp)
        self.cnf.append(clause)

    def add_xor_multi(self, inputs: list, result: int):
        """Add multi-input XOR constraint using auxiliary variables"""
        if len(inputs) == 1:
            # result = inputs[0]
            self.cnf.append([result, -inputs[0]])
            self.cnf.append([-result, inputs[0]])
            return

        # Chain XORs using auxiliary variables
        current = inputs[0]
        for i in range(1, len(inputs)):
            if i == len(inputs) - 1:
                # Last XOR outputs to result
                self.add_xor2(current, inputs[i], result)
            else:
                # Intermediate XOR
                aux = self.new_var()
                self.add_xor2(current, inputs[i], aux)
                current = aux

    def add_z_function(
        self, e1: int, e2: int, e3: int, e4: int, e5: int, e6: int
    ) -> int:
        """Add z function constraint and return the result variable"""
        # Create auxiliary variables for all the AND operations
        and_14 = self.new_var()
        and_23 = self.new_var()
        and_25 = self.new_var()
        and_45 = self.new_var()
        and_56 = self.new_var()

        # Add AND constraints
        self.add_and(e1, e4, and_14)
        self.add_and(e2, e3, and_23)
        self.add_and(e2, e5, and_25)
        self.add_and(e4, e5, and_45)
        self.add_and(e5, e6, and_56)

        # Three-way ANDs
        and_134 = self.new_var()
        and_136 = self.new_var()
        and_145 = self.new_var()
        and_236 = self.new_var()
        and_246 = self.new_var()
        and_356 = self.new_var()

        self.add_and_multi([e1, e3, e4], and_134)
        self.add_and_multi([e1, e3, e6], and_136)
        self.add_and_multi([e1, e4, e5], and_145)
        self.add_and_multi([e2, e3, e6], and_236)
        self.add_and_multi([e2, e4, e6], and_246)
        self.add_and_multi([e3, e5, e6], and_356)

        # Four-way ANDs
        and_1234 = self.new_var()
        and_1235 = self.new_var()
        and_1256 = self.new_var()
        and_2346 = self.new_var()

        self.add_and_multi([e1, e2, e3, e4], and_1234)
        self.add_and_multi([e1, e2, e3, e5], and_1235)
        self.add_and_multi([e1, e2, e5, e6], and_1256)
        self.add_and_multi([e2, e3, e4, e6], and_2346)

        # Five and six-way ANDs
        and_12345 = self.new_var()
        and_13456 = self.new_var()

        self.add_and_multi([e1, e2, e3, e4, e5], and_12345)
        self.add_and_multi([e1, e3, e4, e5, e6], and_13456)

        # Create the final XOR result
        result = self.new_var()

        # All inputs to the XOR
        xor_inputs = [
            e1,
            e5,
            e6,
            and_14,
            and_23,
            and_25,
            and_45,
            and_56,
            and_134,
            and_136,
            and_145,
            and_236,
            and_246,
            and_356,
            and_1234,
            and_1235,
            and_1256,
            and_2346,
            and_12345,
            and_13456,
        ]

        # Add the multi-input XOR constraint
        self.add_xor_multi(xor_inputs, result)

        return result

    def get_u_bit(self, u_vars: List[int], s1_vars: List[int], p_index: int) -> int:
        """Get U bit based on p_index"""
        if p_index == 0:
            return s1_vars[-1]
        return u_vars[p_index]

    def encode_lfsr_shift(self, lfsr_vars: list, feedback_positions: list) -> list:
        """Encode LFSR shift with feedback"""

        # Calculate feedback
        feedback = self.new_var()
        self.add_xor_multi([lfsr_vars[pos] for pos in feedback_positions], feedback)

        new_lfsr = [feedback] + lfsr_vars[:-1]
        return new_lfsr

    def encode_f_vector_update(self, f_vars: list) -> list:
        """Encode F-vector update (61-bit LFSR with feedback from positions 0, 1, 2, 5)"""
        return self.encode_lfsr_shift(f_vars, [0, 1, 2, 5])

    def encode_u_vector_update(self, u_vars, t_array, d_array, s1_bit) -> list:
        """Encode U-vector update with shift and 9-bit insertion"""
        ## Right rotation
        new_u_vars = [u_vars[-1]] + u_vars[:-1]
        old_u = new_u_vars
        for j in range(1, 10):
            result_var = self.new_var()
            d_index = d_array[j - 1]
            self.add_xor2(old_u[d_index], t_array[10 - j], result_var)
            new_u_vars[4 * j - 3] = result_var
        new_u_vars[0] = s1_bit
        return new_u_vars

    def encode_single_round(
        self,
        s1_vars: List[int],
        s2_vars: List[int],
        u_vars: List[int],
        f_vector_vars: list,
        start_shift_index: int = 0,
    ) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
        """Encode one complete round of T310 cipher"""
        current_s1 = s1_vars[:]
        current_s2 = s2_vars[:]
        current_u = u_vars[:]
        current_f = f_vector_vars[:]  # Start with initial F-vector
        a_vars = []
        t_array = [self.new_var() for i in range(10)]
        for var in t_array:
            self.cnf.append([-var])
        for outer_round in range(12):
            for inner_round in range(2):
                # t9: Previous t8 XOR u bit from p26
                u_p26 = self.get_u_bit(current_u, current_s1, self.p[28 - 2] - 1)
                new_t9 = self.new_var()
                self.add_xor2(t_array[8], u_p26, new_t9)
                t_array[9] = new_t9

                # t8: Previous t7 XOR Z function from inputs u_p22 to u_p27
                z_inputs = []
                for i in range(22, 28):
                    z_inputs.append(
                        self.get_u_bit(current_u, current_s1, self.p[i - 2] - 1)
                    )
                z_result = self.add_z_function(*z_inputs)
                new_t8 = self.new_var()
                self.add_xor2(t_array[7], z_result, new_t8)
                t_array[8] = new_t8

                # t7: Previous t6 XOR u_p19
                u_p19 = self.get_u_bit(current_u, current_s1, self.p[21 - 2] - 1)
                new_t7 = self.new_var()
                self.add_xor2(t_array[6], u_p19, new_t7)
                t_array[7] = new_t7

                # t6: Previous t5 XOR Z function from inputs u_p15 to u_p20
                z_inputs = []
                for i in range(15, 21):
                    z_inputs.append(
                        self.get_u_bit(current_u, current_s1, self.p[i - 2] - 1)
                    )
                z_result = self.add_z_function(*z_inputs)
                new_t6 = self.new_var()
                self.add_xor2(t_array[5], z_result, new_t6)
                t_array[6] = new_t6

                # t5: Previous t4 XOR u_p12
                u_p12 = self.get_u_bit(current_u, current_s1, self.p[14 - 2] - 1)
                new_t5 = self.new_var()
                self.add_xor2(t_array[4], u_p12, new_t5)
                t_array[5] = new_t5

                # t4: Previous t3 XOR Z function from inputs u_p8 to u_p13
                z_inputs = []
                for i in range(8, 14):
                    z_inputs.append(
                        self.get_u_bit(current_u, current_s1, self.p[i - 2] - 1)
                    )
                z_result = self.add_z_function(*z_inputs)
                new_t4 = self.new_var()
                self.add_xor2(t_array[3], z_result, new_t4)
                t_array[4] = new_t4

                # t3: Previous t2 XOR u_p5
                u_p5 = self.get_u_bit(current_u, current_s1, self.p[7 - 2] - 1)
                new_t3 = self.new_var()
                self.add_xor2(t_array[2], u_p5, new_t3)
                t_array[3] = new_t3

                # t2: Previous t1 XOR Z function from special inputs
                z_inputs = [
                    current_s2[-1],  # s2[119] - last bit
                    self.get_u_bit(current_u, current_s1, self.p[2 - 2] - 1),
                    self.get_u_bit(current_u, current_s1, self.p[3 - 2] - 1),
                    self.get_u_bit(current_u, current_s1, self.p[4 - 2] - 1),
                    self.get_u_bit(current_u, current_s1, self.p[5 - 2] - 1),
                    self.get_u_bit(current_u, current_s1, self.p[6 - 2] - 1),
                ]
                z_result = self.add_z_function(*z_inputs)
                new_t2 = self.new_var()
                self.add_xor2(t_array[1], z_result, new_t2)
                t_array[2] = new_t2
                # t1: Get F bit and update F-vector
                t_array[1] = current_f[0]  # Current F bit

                # Update F-vector for next round
                current_f = self.encode_f_vector_update(current_f)

                # Update U vector with t_array values
                new_u = self.encode_u_vector_update(
                    current_u,
                    t_array,
                    self.d,
                    current_s1[-1],  # s1_bit is the last bit of s1
                )

                # Rotate S1 and S2
                new_s1 = [current_s1[-1]] + current_s1[:-1]
                new_s2 = [current_s2[-1]] + current_s2[:-1]
                """                 new_s1 = [
                    self.new_var(f"s1_{outer_round}_{inner_round}_{i}")
                    for i in range(len(current_s1))
                ]

                new_s2 = [
                    self.new_var(f"s2_{outer_round}_{inner_round}_{i}")
                    for i in range(len(current_s2))
                ]

                # S1 feedback and shift
                s1_feedback = current_s1[-1]
                self.cnf.append([new_s1[0], -s1_feedback])
                self.cnf.append([-new_s1[0], s1_feedback])

                # Shift S1 (right shift)
                for i in range(1, len(current_s1)):
                    self.cnf.append([new_s1[i], -current_s1[i - 1]])
                    self.cnf.append([-new_s1[i], current_s1[i - 1]])

                # S2 feedback and shift
                s2_feedback = current_s2[-1]  # Last bit of S2
                self.cnf.append([new_s2[0], -s2_feedback])
                self.cnf.append([-new_s2[0], s2_feedback])

                # Shift S2 (right shift)
                for i in range(1, len(current_s2)):
                    self.cnf.append([new_s2[i], -current_s2[i - 1]])
                    self.cnf.append([-new_s2[i], current_s2[i - 1]])
                """

                # Update current state
                current_u = new_u
                current_s1 = new_s1
                current_s2 = new_s2

            # After 126 inner rounds, collect the alpha bit from U vector
            a_vars.append(current_u[self.alpha - 1])

        return current_s1, current_s2, current_u, current_f, a_vars

    def encode_srv_shift(self, srv: List[int]) -> List[int]:
        """Encode SRV shift register update"""
        n = len(srv)
        new_srv = [self.new_var(f"srv_{i}") for i in range(n)]

        # Calculate feedback bit: srv[0] XOR srv[2]
        feedback = self.new_var("srv_feedback")
        self.add_xor2(srv[0], srv[2], feedback)

        # Shift left (opposite of what's done in Rust)
        for i in range(n - 1):
            self.cnf.append([-new_srv[i], srv[i + 1]])
            self.cnf.append([new_srv[i], -srv[i + 1]])

        # Set last bit to feedback
        self.cnf.append([-new_srv[n - 1], feedback])
        self.cnf.append([new_srv[n - 1], -feedback])

        return new_srv

    def encode_encrypt_character(
        self, a_vars: List[int], plaintext_bits: List[int]
    ) -> List[int]:
        """Encode the encrypt_character function for a 5-bit character"""
        # Create variables for SRV_2 and SRV_3
        srv_2 = [self.new_var(f"srv2_{i}") for i in range(5)]
        srv_3 = [self.new_var(f"srv3_{i}") for i in range(5)]

        # Initialize SRV_2 with first 5 bits from a_vars
        for i in range(5):
            self.cnf.append([-srv_2[i], a_vars[i]])
            self.cnf.append([srv_2[i], -a_vars[i]])

        # Initialize SRV_3 with all ones
        for i in range(5):
            self.cnf.append([srv_3[i]])

        # Encode the shift loop
        # We'll need to unroll the loop since we don't know how many iterations will be needed
        # For SAT encoding, we'll use a fixed number of shifts, e.g., 16 (enough to cover all possible cases)
        max_shifts = 16
        all_srv2 = [srv_2]
        all_srv3 = [srv_3]

        for shift in range(max_shifts):
            new_srv2 = self.encode_srv_shift(all_srv2[-1])
            new_srv3 = self.encode_srv_shift(all_srv3[-1])
            all_srv2.append(new_srv2)
            all_srv3.append(new_srv3)

        # Check the all-True or all-False condition for each shift
        stop_conditions = []
        for shift in range(max_shifts):
            # All zeros condition for SRV_2
            all_zeros = self.new_var(f"srv2_all_zeros_{shift}")
            zero_clauses = []
            for bit in all_srv2[shift]:
                zero_clauses.append(-bit)
            self.add_and_multi(zero_clauses, all_zeros)

            # All ones condition for SRV_2
            all_ones = self.new_var(f"srv2_all_ones_{shift}")
            one_clauses = []
            for bit in all_srv2[shift]:
                one_clauses.append(bit)
            self.add_and_multi(one_clauses, all_ones)

            # Stop condition is either all zeros or all ones
            stop = self.new_var(f"stop_{shift}")
            self.add_xor2(all_zeros, all_ones, stop)
            stop_conditions.append(stop)

        # Create variables for the final SRV_3 state
        final_srv3 = [self.new_var(f"final_srv3_{i}") for i in range(5)]

        # Encode the condition to select the final SRV_3 based on stop condition
        for shift in range(max_shifts):
            for i in range(5):
                # If this shift is the first to satisfy the stop condition,
                # then final_srv3[i] = all_srv3[shift][i]
                # We need a variable to indicate "this is the first shift to stop"
                first_stop = self.new_var(f"first_stop_{shift}")

                # This shift stops AND all previous shifts don't stop
                prev_stops = []
                for prev in range(shift):
                    prev_stops.append(stop_conditions[prev])

                if prev_stops:
                    not_prev_stop = self.new_var(f"not_prev_stop_{shift}")
                    # NOT of all previous stops combined
                    for prev_stop in prev_stops:
                        self.cnf.append([-not_prev_stop, -prev_stop])
                    # If any prev_stop is True, not_prev_stop is False
                    all_neg_prev = [-p for p in prev_stops]
                    all_neg_prev.append(not_prev_stop)
                    self.cnf.append(all_neg_prev)

                    # first_stop = stop_conditions[shift] AND not_prev_stop
                    self.add_and(stop_conditions[shift], not_prev_stop, first_stop)
                else:
                    # For the first shift, first_stop is just stop_conditions[0]
                    self.cnf.append([-first_stop, stop_conditions[shift]])
                    self.cnf.append([first_stop, -stop_conditions[shift]])

                # If this is the first stop, final_srv3[i] = all_srv3[shift][i]
                self.cnf.append([-first_stop, -final_srv3[i], all_srv3[shift][i]])
                self.cnf.append([-first_stop, final_srv3[i], -all_srv3[shift][i]])

        # Now handle the XOR with a_vars[5:10] and plaintext_bits
        ciphertext_bits = [self.new_var(f"ciphertext_{i}") for i in range(5)]
        for i in range(5):
            # ciphertext_bits[i] = final_srv3[i] XOR a_vars[5+i]
            temp_xor = self.new_var(f"temp_xor_{i}")
            self.add_xor2(final_srv3[i], a_vars[5 + i], temp_xor)

            # Apply plaintext XOR for end result
            self.add_xor2(temp_xor, plaintext_bits[i], ciphertext_bits[i])

        return ciphertext_bits

    def encode_encrypt_character_simple(
        self, a_vars: List[int], plaintext_bits: List[int], pair_index: int
    ) -> List[int]:
        """Encode a simplified version of encrypt_character with proper variable indexing"""
        # Extract first 5 bits from a_vars for keystream
        keystream = a_vars[:5]

        # Create ciphertext bits through XOR - now using pair_index in variable names
        ciphertext_bits = [
            self.new_var(f"ciphertext_{pair_index}_{i}") for i in range(5)
        ]
        for i in range(5):
            self.add_xor2(keystream[i], plaintext_bits[i], ciphertext_bits[i])

        return ciphertext_bits

    def setup_known_plaintext_attack(
        self, plaintext_bits: List[List[bool]], ciphertext_bits: List[List[bool]]
    ) -> Tuple[List[int], List[int]]:
        """Set up a known plaintext attack with given plaintext and ciphertext"""
        assert len(plaintext_bits) == len(ciphertext_bits), (
            "Plaintext and ciphertext must have the same length"
        )

        # Create variables for the key
        s1_vars = [self.new_var(f"key_s1_{i}") for i in range(120)]
        s2_vars = [self.new_var(f"key_s2_{i}") for i in range(120)]

        # Create F-vector variables (can be fixed or unknown)
        f_vector_vars = [self.new_var(f"f_vector_{i}") for i in range(61)]

        # Create initial U-vector variables (standard U-vector)
        u_vars = [self.new_var(f"u_{i}") for i in range(37)]

        # Set constraints for standard U-vector
        for i, bit in enumerate(self.standard_u_vector[:37]):
            if bit:
                self.cnf.append([u_vars[i]])
            else:
                self.cnf.append([-u_vars[i]])

        # Add parity constraints for the key (each 24-bit block must have odd parity)
        """ 
        for i in range(5):
            block_start = i * 24
            block_end = block_start + 24

            # S1 parity bit
            s1_parity_vars = s1_vars[block_start:block_end]
            s1_parity = self.new_var(f"s1_parity_{i}")
            self.add_xor_multi(s1_parity_vars, s1_parity)
            # Ensure odd parity (parity bit must be 1)
            self.cnf.append([s1_parity])

            # S2 parity bit
            s2_parity_vars = s2_vars[block_start:block_end]
            s2_parity = self.new_var(f"s2_parity_{i}")
            self.add_xor_multi(s2_parity_vars, s2_parity)
            # Ensure odd parity (parity bit must be 1)
            self.cnf.append([s2_parity])
    """
        # For each plaintext-ciphertext pair
        current_s1 = s1_vars
        current_s2 = s2_vars
        current_u = u_vars
        current_f = f_vector_vars

        for pair_index, (pt_bits, ct_bits) in enumerate(
            zip(plaintext_bits, ciphertext_bits)
        ):
            # Convert boolean lists to variable lists
            pt_vars = [self.new_var(f"pt_{pair_index}_{i}") for i in range(5)]
            ct_vars = [self.new_var(f"ct_{pair_index}_{i}") for i in range(5)]

            # Set the plaintext and ciphertext bits
            for i in range(5):
                if pt_bits[i]:
                    self.cnf.append([pt_vars[i]])
                else:
                    self.cnf.append([-pt_vars[i]])

                if ct_bits[i]:
                    self.cnf.append([ct_vars[i]])
                else:
                    self.cnf.append([-ct_vars[i]])
            print("round : " + str(pair_index))
            # Run the cipher for one round to get a_vars

            current_s1, current_s2, current_u, current_f, a_vars = (
                self.encode_single_round(current_s1, current_s2, current_u, current_f)
            )

            # Encrypt the plaintext to get calculated ciphertext
            # calc_ct_vars = self.encode_encrypt_character(a_vars, pt_vars)
            calc_ct_vars = self.encode_encrypt_character_simple(
                a_vars, pt_vars, pair_index
            )

            # Add constraint that calculated ciphertext matches actual ciphertext
            for i in range(5):
                self.cnf.append([-calc_ct_vars[i], ct_vars[i]])
                self.cnf.append([calc_ct_vars[i], -ct_vars[i]])

        return s1_vars, s2_vars, f_vector_vars

    def recover_key(
        self, plaintext_bits: List[List[bool]], ciphertext_bits: List[List[bool]]
    ) -> Tuple[List[bool], List[bool]]:
        """Recover the key using the known plaintext attack"""
        # Set up the attack
        s1_vars, s2_vars, f_vars = self.setup_known_plaintext_attack(
            plaintext_bits, ciphertext_bits
        )

        # Create a solver
        solver = Cadical195()
        # solver = Glucose42()

        # Add the CNF clauses to the solver
        for clause in self.cnf:
            solver.add_clause(clause)

        print("starting solving")
        # Solve the SAT problem
        if solver.solve():
            # Extract the solution
            model = solver.get_model()

            # Extract the key bits
            s1_key = [model[var - 1] > 0 for var in s1_vars]
            s2_key = [model[var - 1] > 0 for var in s2_vars]
            iv_key = [model[var - 1] > 0 for var in f_vars]
            return s1_key, s2_key, iv_key
        else:
            raise ValueError(
                "No solution found - either the ciphertext doesn't match the plaintext with any key, or there's an error in the encoding."
            )


def parse_and_convert_to_bit_chunks(
    boolean_string: str, chunk_size: int = 5
) -> List[List[bool]]:
    """
    Parse a string representation of a boolean list and convert it into chunks of a specified size.

    Args:
        boolean_string (str): The string representation of a boolean list (e.g., "[True, False, True]").
        chunk_size (int): The size of each chunk (default is 5).

    Returns:
        List[List[bool]]: A list of chunks, where each chunk is a list of booleans.
    """
    # Convert the string to a Python list of booleans
    boolean_list = ast.literal_eval(
        boolean_string.replace("true", "True").replace("false", "False")
    )

    # Split the list into chunks
    return [
        boolean_list[i : i + chunk_size]
        for i in range(0, len(boolean_list), chunk_size)
    ]


def main():
    # Initialize the CCITT2 codec
    codec = SimpleCCITT2.SimpleCCITT2()

    # Convert to bits

    plaintext_bits = parse_and_convert_to_bit_chunks(
        "[true, false, true, false, false, false, false, false, false, true, true, false, false, true, false, true, false, false, true, false, true, true, false, false, false]"
    )

    ciphertext_bits = parse_and_convert_to_bit_chunks(
        "[true, false, false, false, false, true, false, true, true, false, true, false, false, true, false, false, true, false, false, false, true, true, false, true, false]"
    )

    # Initialize SAT solver
    solver = T310SatSolver()

    # Recover the key
    try:
        s1_key, s2_key, iv = solver.recover_key(plaintext_bits, ciphertext_bits)

        # Convert keys to BitVec format for verification

        s1_bitvec = bitvec_from_bools(s1_key)
        s2_bitvec = bitvec_from_bools(s2_key)
        iv_bitvec = bitvec_from_bools(iv)

        print("Recovered S1 key:", s1_key)
        print("Recovered S1 key:", s1_bitvec)
        print("Recovered S2 key:", s2_key)
        print("Recovered S2 key:", s2_bitvec)
        print("Recovered IV key:", iv)
        print("Recovered IV key:", iv_bitvec)

        # Verify the solution
        print("Verifying solution... Not yet implemented")

    except ValueError as e:
        print(f"Error: {e}")


# Helper function to split a list into chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# Helper function to create a BitVec from a list of booleans
def bitvec_from_bools(bools):
    """Convert a list of booleans to a BitVec string representation."""
    return "".join("1" if b else "0" for b in bools)


if __name__ == "__main__":
    main()
