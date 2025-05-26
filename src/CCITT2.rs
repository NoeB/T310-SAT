use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
// Simple error type
#[derive(Debug)]
pub enum SimpleError {
    CharacterNotFound(char, bool), // (char, is_figure_mode)
    InvalidCode(String),
    InvalidLength,
}
impl From<SimpleError> for PyErr {
    fn from(error: SimpleError) -> Self {
        match error {
            SimpleError::CharacterNotFound(c, is_figure_mode) => PyValueError::new_err(format!(
                "Character not found: '{}' (in {} mode)",
                c,
                if is_figure_mode { "figures" } else { "letters" }
            )),
            SimpleError::InvalidCode(code) => {
                PyValueError::new_err(format!("Invalid code: {}", code))
            }
            SimpleError::InvalidLength => {
                PyValueError::new_err("Binary string length must be multiple of 5")
            }
        }
    }
}
#[pyclass]
pub struct SimpleCCITT2 {
    // Maps characters to 5-bit codes (as binary strings)
    ltrs_map: HashMap<char, String>,
    figs_map: HashMap<char, String>,

    // Maps 5-bit codes (as binary strings) to characters
    code_to_ltrs: HashMap<String, char>,
    code_to_figs: HashMap<String, char>,

    // Special control codes
    ltrs_code: String, // Code to switch to letters mode
    figs_code: String, // Code to switch to figures mode
}
#[pymethods]
impl SimpleCCITT2 {
    #[new]
    pub fn new() -> Self {
        let mut codec = Self {
            ltrs_map: HashMap::new(),
            figs_map: HashMap::new(),
            code_to_ltrs: HashMap::new(),
            code_to_figs: HashMap::new(),
            ltrs_code: "11111".to_string(),
            figs_code: "11011".to_string(),
        };

        codec.init_mappings();
        codec
    }

    pub fn py_encode(&self, text: &str) -> PyResult<String> {
        self.encode(text).map_err(PyErr::from)
    }

    pub fn py_decode(&self, binary: &str) -> PyResult<String> {
        self.decode(binary).map_err(PyErr::from)
    }

    pub fn py_encode_to_bools(&self, text: &str) -> PyResult<Vec<bool>> {
        self.encode_to_bools(text).map_err(PyErr::from)
    }

    pub fn py_decode_from_bools(&self, bools: Vec<bool>) -> PyResult<String> {
        self.decode_from_bools(&bools).map_err(PyErr::from)
    }
}
impl SimpleCCITT2 {
    //T-310 is used the ITA2- CCITT2 code
    // Mapping: https://scz.bplaced.net/t310-fs.html#t310-4

    fn init_mappings(&mut self) {
        let letter_mappings = [
            ("00011", 'A'),
            ("11001", 'B'),
            ("01110", 'C'),
            ("01001", 'D'),
            ("00001", 'E'),
            ("01101", 'F'),
            ("11010", 'G'),
            ("10100", 'H'),
            ("00110", 'I'),
            ("01011", 'J'),
            ("01111", 'K'),
            ("10010", 'L'),
            ("11100", 'M'),
            ("01100", 'N'),
            ("11000", 'O'),
            ("10110", 'P'),
            ("10111", 'Q'),
            ("01010", 'R'),
            ("00101", 'S'),
            ("10000", 'T'),
            ("00111", 'U'),
            ("11110", 'V'),
            ("10011", 'W'),
            ("11101", 'X'),
            ("10101", 'Y'),
            ("10001", 'Z'),
            ("00100", ' '),
            ("01000", '\n'),
            ("00010", '\r'),
        ];
        // Figures mode mappings
        let figure_mappings = [
            ("00011", '-'),
            ("11001", '?'),
            ("01110", ':'),
            ("01001", '¶'), //wer da?`
            ("00001", '3'),
            ("01101", 'º'), //unused
            ("11010", 'º'), //unused
            ("10100", 'º'), //unused
            ("00110", '8'),
            ("01011", '⍾'), // Bell
            ("01111", '('),
            ("10010", ')'),
            ("11100", '.'),
            ("01100", ','),
            ("11000", '9'),
            ("10110", '0'),
            ("10111", '1'),
            ("01010", '4'),
            ("00101", '\''),
            ("10000", '5'),
            ("00111", '7'),
            ("11110", '='),
            ("10011", '2'),
            ("11101", '/'),
            ("10101", '6'),
            ("10001", '+'),
            ("00100", ' '),
            ("01000", '\n'),
            ("00010", '\r'),
        ];

        // Fill the maps
        for (code, ch) in letter_mappings {
            self.ltrs_map.insert(ch, code.to_string());
            self.code_to_ltrs.insert(code.to_string(), ch);
        }

        for (code, ch) in figure_mappings {
            self.figs_map.insert(ch, code.to_string());
            self.code_to_figs.insert(code.to_string(), ch);
        }
    }

    pub fn encode(&self, text: &str) -> Result<String, SimpleError> {
        let mut result = String::new();
        let mut in_figure_mode = false;

        for c in text.chars() {
            if in_figure_mode {
                // Check if character exists in figures mode
                if let Some(code) = self.figs_map.get(&c) {
                    result.push_str(code);
                    continue;
                }

                // If not, check if it's in letters mode
                if let Some(code) = self.ltrs_map.get(&c.to_ascii_uppercase()) {
                    // Switch to letters mode
                    result.push_str(&self.ltrs_code);
                    result.push_str(code);
                    in_figure_mode = false;
                } else {
                    return Err(SimpleError::CharacterNotFound(c, in_figure_mode));
                }
            } else {
                // Check if character exists in letters mode
                if let Some(code) = self.ltrs_map.get(&c.to_ascii_uppercase()) {
                    result.push_str(code);
                    continue;
                }

                // If not, check if it's in figures mode
                if let Some(code) = self.figs_map.get(&c) {
                    // Switch to figures mode
                    result.push_str(&self.figs_code);
                    result.push_str(code);
                    in_figure_mode = true;
                } else {
                    return Err(SimpleError::CharacterNotFound(c, in_figure_mode));
                }
            }
        }

        Ok(result)
    }

    pub fn decode(&self, binary: &str) -> Result<String, SimpleError> {
        if binary.len() % 5 != 0 {
            return Err(SimpleError::InvalidLength);
        }

        let mut result = String::new();
        let mut in_figure_mode = false;

        for i in (0..binary.len()).step_by(5) {
            let code = &binary[i..i + 5];

            // Check for mode shifts
            if code == self.figs_code {
                in_figure_mode = true;
                continue;
            } else if code == self.ltrs_code {
                in_figure_mode = false;
                continue;
            }

            if in_figure_mode {
                if let Some(&ch) = self.code_to_figs.get(code) {
                    result.push(ch);
                } else {
                    return Err(SimpleError::InvalidCode(code.to_string()));
                }
            } else {
                if let Some(&ch) = self.code_to_ltrs.get(code) {
                    result.push(ch);
                } else {
                    return Err(SimpleError::InvalidCode(code.to_string()));
                }
            }
        }

        Ok(result)
    }

    pub fn encode_to_bools(&self, text: &str) -> Result<Vec<bool>, SimpleError> {
        let binary = self.encode(text)?;
        Ok(binary.chars().map(|c| c == '1').collect())
    }

    pub fn decode_from_bools(&self, bools: &[bool]) -> Result<String, SimpleError> {
        let binary: String = bools.iter().map(|&b| if b { '1' } else { '0' }).collect();
        self.decode(&binary)
    }
}

#[pymodule]
#[pyo3(name = "SimpleCCITT2")]
fn simple_ccitt2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SimpleCCITT2>()?;
    Ok(())
}
