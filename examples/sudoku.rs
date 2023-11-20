//! # Example: Sudoku as Exact Cover
//!
//! [Sudoku] can be solved as a pure exact cover problem. There are 729 options
//! that correspond to the placement of 9 digits in 81 cells. There are 324
//! items in 4 categories of 81: each of the 81 cells must have 1 digit, each of
//! the 9 rows must have exactly 1 of each of the 9 digits, each of the 9
//! columns must have exactly 1 of each of the 9 digits, and each of the 9 3x3
//! boxes must have exactly 1 of each of the 9 digits. Each option thus covers
//! exactly 4 items.
//!
//! [Sudoku]: <https://en.wikipedia.org/wiki/Sudoku>

use xcov::{DlxBuilder, ExactCoverProblem, MrvExactCoverSearch};

/// Solves a standard sudoku puzzle given as an 81 character `str` (starting
/// with the top left cell and listed in row-major order). Any character other
/// than 1-9 is interpreted as a blank.
#[must_use]
pub fn solve(puzzle: &str) -> Option<String> {
    if puzzle.len() != 81 {
        return None;
    }

    let mut builder = DlxBuilder::new(9 * 9 * 4, 0);
    let mut givens = IntSet::<{ (324 + 63) / 64 }>::new();

    // add puzzle givens
    for (i, ch) in puzzle.chars().enumerate() {
        let Some(digit) = ch.to_digit(10) else {
            continue;
        };
        if !(1..=9).contains(&digit) {
            continue;
        }
        let digit = (digit - 1) as usize;
        let option = sudoku_items(i, digit);
        builder.add_option(&option);
        for &item in &option {
            givens.insert(item);
        }
    }

    // add all options that do not conflict with givens
    for cell in 0..81 {
        for digit in 0..9 {
            let option = sudoku_items(cell, digit);
            if option.iter().any(|&i| givens.get(i)) {
                continue;
            }
            builder.add_option(&option);
        }
    }

    let mut ec = MrvExactCoverSearch::new(builder.build());
    ec.search();

    let solution = ec.current_solution()?;
    let mut result = [' '; 81];
    for option in solution {
        let Ok(option): Result<[usize; 4], _> = option.try_into() else {
            unreachable!()
        };
        let [cell, digit] = sudoku_invert_items(&option);
        let Ok(digit): Result<u32, _> = digit.try_into() else {
            unreachable!()
        };
        result[cell] = char::from_digit(digit + 1, 10)?;
    }
    Some(result.iter().collect())
}

/// Converts a sudoku cell and digit into its 4 items
fn sudoku_items(cell: usize, digit: usize) -> [usize; 4] {
    let row = cell / 9;
    let col = cell % 9;
    [
        1 + cell,
        1 + 81 + row * 9 + digit,
        1 + 81 * 2 + col * 9 + digit,
        1 + 81 * 3 + (row / 3 * 3 + col / 3) * 9 + digit,
    ]
}

/// Converts 4 items into a sudoku cell and digit
fn sudoku_invert_items(items: &[usize; 4]) -> [usize; 2] {
    let digit = (items[1] - 81 - 1) % 9;
    [items[0] - 1, digit]
}

/// A set of integers with fixed maximum
struct IntSet<const N: usize> {
    bits: [u64; N],
}

impl<const N: usize> IntSet<N> {
    fn new() -> Self {
        Self { bits: [0; N] }
    }

    fn coords(n: usize) -> (usize, u32) {
        let i = n / 64;
        assert!(i < N);
        let sh = (n % 64).try_into().expect("mod 64 fits in u32");
        (i, sh)
    }

    fn insert(&mut self, n: usize) {
        let (i, sh) = Self::coords(n);
        self.bits[i] |= 1 << sh;
    }

    fn get(&self, n: usize) -> bool {
        let (i, sh) = Self::coords(n);
        self.bits[i] & 1 << sh != 0
    }
}

fn main() {
    let Some(puzzle) = std::env::args().nth(1) else {
        panic!("Missing puzzle argument")
    };
    if let Some(solution) = solve(&puzzle) {
        println!("{solution}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn puzzle1() {
        let solution = solve(
            "769000028000400009000000005005000000090860070280003000008300091002080600000000200",
        )
        .unwrap();
        let expected =
            "769531428521478369834296715175942836493865172286713954648327591352189647917654283";
        assert_eq!(solution, expected);
    }
}
