# Exact Cover

This crate provides an implementation of Knuth's [Algorithm X] for solving
[exact cover problems].

The algorithms and data structures in this module, as well as the
terminology of "options" and "items", come from Donald Knuth's <cite>The Art
of Computer Programming, section 7.2.2</cite>,
<https://www-cs-faculty.stanford.edu/~knuth/fasc5c.ps.gz>.

[Algorithm X]: https://en.wikipedia.org/wiki/Knuth%27s_Algorithm_X
[exact cover problem]: https://en.wikipedia.org/wiki/Exact_cover

* No dependencies
* 100% safe Rust
* Choose whether to find a single solution or all possible solutions
* Designed for flexibility (see below)

From a high level, Algorithm X is simply a backtracking exhaustive search
algorithm. At a lower level, Algorithm X cleverly manipulates linked lists using
the "dancing links" technique to make the backtracking more efficient. This
crate implements the backtracking search as a provided trait method
[`ExactCoverProblem::search`], while the dancing links data structure
and operations are implemented by the [`Dlx`] struct. A custom implementation of
the `ExactCoverProblem` trait can make use of the dancing links structure
provided by `Dlx`, control the order items are selected for covering, and apply
additional filters to options for problems beyond pure exact cover.

A ready-to-use implementation using the *minimum remaining values* heuristic to
select items (i.e. select the item covered by the fewest number of available
options) is provided with [`MrvExactCoverSearch`]. Its code can be helpful as a
starting point for custom implementations.

## Examples

The following pure exact cover problem is often used as an example in texts by
Knuth, and consists of 7 items and 6 options. For a more practical application,
see `examples/sudoku.rs`.

```rust
# use xcov::*;
// the following example has exactly one solution
let mut dlx = DlxBuilder::new(7, 0);
dlx.add_option(&[      3,    5      ]);
dlx.add_option(&[1,       4,       7]);
dlx.add_option(&[   2, 3,       6   ]);
dlx.add_option(&[1,       4,    6   ]);
dlx.add_option(&[   2,             7]);
dlx.add_option(&[         4, 5,    7]);

let mut problem = MrvExactCoverSearch::new(dlx.build());
problem.search();
let solution = problem.current_solution().unwrap().into_iter().collect::<Vec<_>>();
assert_eq!(
   solution,
   [
      &[1, 4, 6][..],
      &[2, 7],
      &[3, 5],
   ]
);

// can look for additional solutions but this problem has none
assert!(!problem.search());
assert!(problem.current_solution().is_none());
```
