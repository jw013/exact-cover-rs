# Exact Cover

An *[exact cover problem]* is a mathematical problem that consists of a universe
of "items" and a set of "options" which are sets of items. A solution to the
problem is a set of pairwise disjoint options whose union is the universe of
items. In other words, a solution must include each item in exactly one option.
As an extension, exact cover problems can also have secondary items which do not
need to be covered in a valid solution but if included can be covered at most
once.

Any problem that involves placing a finite set of objects in a finite set of
possible positions under some non-overlapping constraint can likely be reduced
to (or formulated in terms of) an exact cover problem. Every position that can
be occupied can be modeled as an "item", and every possible placement of an
object is an "option" that covers the set of items (positions) it occupies. One
example is the [N Queens] puzzle, which tries to place 8 queens on a standard
8x8 chessboard such that no two queens can block each other's movement. As an
exact cover problem, it consists of 64 options (squares on which queens can be
placed), each covering the rank, file, and diagonals that square belongs to.
Another example is [Sudoku], in which each cell in a 9x9 grid must be filled
with one of the numbers 1-9 such that no column, row, or 3x3 sub-grid contains
two of the same number. Any placement of a number in a cell corresponds to an
option covering 4 items: 1 item for filling that specific cell with any number,
and 3 separate items for filling that row, column, and 3x3 box with that
specific number.

Donald Knuth's Algorithm X can find all solutions of any exact cover
problem. From a high level, Algorithm X is simply a backtracking exhaustive
search algorithm. At a lower level, Algorithm X cleverly manipulates linked
lists to make the backtracking more efficient. This crate implements
Algorithm X but separates the two levels: the backtracking search is
implemented by the [`ExactCoverProblem`] trait, while the lower level linked
list data structure and operations are implemented by the [`Dlx`] struct.
This split arrangement gives the implementor more flexibility to customize
the search algorithm for the specific problem at hand. A ready-to-use
implementation of `ExactCoverProblem` built on `Dlx` can be found at
[`MrvExactCoverSearch`], and its code can be helpful as a starting point for
custom implementations.

The algorithms and data structures in this module, as well as the
terminology of "options" and "items", come from Donald Knuth's <cite>The Art
of Computer Programming, section 7.2.2</cite>,
<https://www-cs-faculty.stanford.edu/~knuth/fasc5c.ps.gz>.

[exact cover problem]: https://en.wikipedia.org/wiki/Exact_cover
[N Queens]: https://en.wikipedia.org/wiki/Eight_queens_puzzle
[Sudoku]: https://en.wikipedia.org/wiki/Sudoku

## Quick Start Roadmap

1. Map your problem to an exact cover problem (i.e. decide what the items and
   options will be).
2. Use [`DlxBuilder::new`] and [`DlxBuilder::add_option`] to add all possible
   options.
3. When finished building, call [`DlxBuilder::build`] and pass the result to
   [`MrvExactCoverSearch::new`].
4. Call [`MrvExactCoverSearch::search`].
5. Check if a solution was found with
   [`MrvExactCoverSearch::current_solution`].
6. The solution is simply a list of options and will need to be mapped back
   into a more usable form for your problem (i.e. the reverse of step 1).
7. _Optional_: if the last `search` returned true, steps 4-6 can be repeated
   to find more solutions.
