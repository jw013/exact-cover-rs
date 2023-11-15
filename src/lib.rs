#![doc = include_str!("../README.md")]
#![warn(clippy::pedantic)]
#![deny(rustdoc::broken_intra_doc_links, unsafe_code)]

/// A generalized exact cover problem. See [module](self) documentation for additional details.
pub trait ExactCoverProblem {
    /// Chooses the next item to cover, covers it and returns true. Returns false if there are no
    /// items left to cover (indicating a solution has been found).
    ///
    /// "Covering an item" means to remove the item from the set of remaining items to cover and to
    /// mark all options that include the item as unavailable for covering other items.
    ///
    /// This method is intended for use by [`search`](Self::search) and is not meant to be called
    /// manually.
    ///
    /// # Item Selection
    ///
    /// A naive exhaustive search could find all solutions by simply selecting options one at a
    /// time, branching at each step to explore all possibilities. The backtracking `search`
    /// algorithm is a bit more clever --- it alternates between choosing (without branching)[^1] an
    /// item to cover and branching on options that cover the selected item. This approach still
    /// finds every possible solution but with two benefits over the naive approach: (1) it reduces
    /// the branching factor (number of possible alternatives) since the set of options that cover
    /// any particular item is usually much smaller than the set of all remaining options, and (2)
    /// it can detect a dead-end as soon as any item has no remaining options to cover it, and does
    /// not have to wait until it runs out of all options.
    ///
    /// The branching factor can be reduced by selecting items with few choices for options. Knuth
    /// calls this the *minimum remaining values (MRV)* heuristic. Reducing the branching factor
    /// generally results in smaller search trees but there may exist exact cover problems where
    /// another item-selection strategy is more efficient than MRV (e.g. by trading increased
    /// branching factor for reduced search depth).
    ///
    /// The item selection strategy can be customized by implementations of this method. See
    /// [`MrvExactCoverSearch`] for an implementation using the MRV heuristic.
    ///
    /// [^1]: It is not necessary to branch on the choice of item. All required items will
    /// eventually be covered by any valid solution, so any sequence of item choices will still find
    /// all solutions.
    fn try_next_item(&mut self) -> bool;

    /// Selects an option that covers the current item (i.e. the item selected by the most recent
    /// call to [`try_next_item`](Self::try_next_item)) to add to the current candidate solution,
    /// then returns true. If there are no more options to try (indicating a dead end and a need to
    /// backtrack), undoes the most recent `try_next_item` and returns false.
    ///
    /// "Selecting an option" means to cover every item included in the selected option (except for
    /// the ones that are already covered).
    ///
    /// This method is intended for use by [`search`](Self::search) and is not meant to be called
    /// manually.
    ///
    /// Implementors should track which options have already been tried for the current item to
    /// avoid redundant computations and infinite loops. Selecting an option that does not cover the
    /// current item will likely cause the search algorithm to return garbage results.
    ///
    /// Implementors are free to filter out options using any additional criteria, which can be
    /// useful for solving problems that are like exact cover with some extra constraints that
    /// cannot be captured as items.
    fn select_option_or_undo_item(&mut self) -> bool;

    /// Undoes the most recent "select option" operation done by
    /// [`select_option_or_undo_item`](Self::select_option_or_undo_item) and returns true, or does
    /// nothing and returns false if there is nothing to undo, which indicates either that a search
    /// has exhausted the all possibilities or that searching has not yet started (due to the
    /// backtracking nature of the search algorithm, the two conditions are indistinguishable).
    ///
    /// This method is intended for use by [`search`](Self::search) and is not meant to be called
    /// manually.
    fn try_undo_option(&mut self) -> bool;

    /// Searches for a solution and returns when one is found or when the entire
    /// search tree has been exhausted, whichever comes first. Returns `true` if
    /// there is more searching to do, `false` otherwise.
    ///
    /// As long as it returns `true`, this method may be called again to look
    /// for additional solutions --- each call will resume searching from where
    /// the last call returned, so long as the underlying type is not mutated in
    /// any other way in between.
    ///
    /// **Note**: it may seem like the return value indicates whether a solution
    /// was found and for the most part this does work, except in the degenerate
    /// case where the problem being solved has no items. Then, the empty set of
    /// no options will be found as the only valid solution and `false` will be
    /// returned at the same time because no possible additional solutions are
    /// possible.
    fn search(&mut self) -> bool {
        // todo: consider possibility of adding profiling and instrumentation like Knuth's version,
        //   e.g. progress indicator, counting of operations (mems), giving up if no solution found
        //   within time limit.

        if self.try_undo_option() {
            // goto 'RESUME; // ... but since that is not allowed, duplicate enough of the relevant
            // code here until we can rejoin the main loop
            while !self.select_option_or_undo_item() {
                if !self.try_undo_option() {
                    return false;
                }
            }
            if !self.try_next_item() {
                return true;
            }
        } else {
            // start search on a fresh problem
            if !self.try_next_item() {
                // empty problem with no items
                return false;
            }
        }

        loop {
            // recursion loop; expects a successful try_next_item to have already occurred
            while self.select_option_or_undo_item() {
                if !self.try_next_item() {
                    return true;
                }
            }
            // unwind / branch loop
            loop {
                if !self.try_undo_option() {
                    return false;
                }
                // 'RESUME:
                if self.select_option_or_undo_item() {
                    break;
                }
            }
            if !self.try_next_item() {
                return true;
            }
        }
    }
}

struct DoubleIndexLink {
    prev: usize,
    next: usize,
}

trait DoubleIndexLinkedList {
    fn remove_links(&mut self, target: usize);
    fn restore_links(&mut self, target: usize);
    fn is_removed(&self, target: usize) -> bool;
}

impl<T> DoubleIndexLinkedList for T
where
    T: core::ops::IndexMut<usize, Output = DoubleIndexLink>,
{
    fn remove_links(&mut self, target: usize) {
        let DoubleIndexLink { prev, next } = self[target];
        self[prev].next = next;
        self[next].prev = prev;
    }

    fn restore_links(&mut self, target: usize) {
        let DoubleIndexLink { prev, next } = self[target];
        self[prev].next = target;
        self[next].prev = target;
    }

    fn is_removed(&self, target: usize) -> bool {
        let DoubleIndexLink { prev, next } = self[target];
        !(self[prev].next == target && self[next].prev == target)
    }
}

/// Created by [`Dlx::primary_items`]
struct LinkIterator<'a> {
    list: &'a [DoubleIndexLink],
    head: usize,
    cursor: usize,
}

impl<'a> Iterator for LinkIterator<'a> {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.cursor).filter(|&c| c != self.head).map(|c| {
            self.cursor = self.list[c].next;
            c
        })
    }
}

impl<'a> LinkIterator<'a> {
    fn from_slice(slice: &'a [DoubleIndexLink], head: usize) -> Self {
        Self {
            list: slice,
            head,
            cursor: slice[head].next,
        }
    }
}

/// Opaque handle used by [`Dlx`] to refer to an option.
///
/// Handles are not transferrable across different `Dlx` instances and should be dropped if the
/// instance they came from is mutated.
#[derive(Copy, Clone)]
pub struct DlxOption(usize);

/// Builder for [`Dlx`]
pub struct DlxBuilder {
    dlx: Dlx,
}

impl DlxBuilder {
    /// Creates an empty Dlx instance with the specified number of primary and
    /// secondary items but no options. Primary items are automatically numbered
    /// starting at 1 and secondary item numbers follow consecutively after
    /// primary items.
    ///
    /// # Panics
    ///
    /// If too many items (`n_primary + n_secondary`) exceeds memory or `usize`
    /// limitations.
    #[must_use]
    pub fn new(n_primary: usize, n_secondary: usize) -> Self {
        let n = n_primary.checked_add(n_secondary).expect("Too many items");
        let len = n.checked_add(2).expect("Too many items");
        let mut dlx = Dlx {
            h_links: Vec::with_capacity(len),
            v_links: Vec::with_capacity(len),
            data: vec![0; len],
            selected_options: Vec::new(),
            current_item: None,
        };

        // Initialize hlinks to two separate circular doubly linked lists:
        // primary items list has 0 as its header node; secondary items list header is at n + 1
        dlx.h_links.push(DoubleIndexLink {
            prev: n_primary,
            next: 1,
        });
        for i in 1..=n {
            dlx.h_links.push(DoubleIndexLink {
                prev: i - 1,
                next: i + 1,
            });
        }
        dlx.h_links.push(DoubleIndexLink {
            prev: n,
            next: n_primary + 1,
        });
        dlx.h_links[n_primary].next = 0;
        dlx.h_links[n_primary + 1].prev = n + 1;

        // item (column) headers
        // 0-th entry is unused
        for i in 0..=n {
            dlx.v_links.push(DoubleIndexLink { prev: i, next: i });
        }
        // first row spacer node
        dlx.v_links.push(DoubleIndexLink { prev: 0, next: 0 });
        Self { dlx }
    }

    /// Adds option represented as a sorted (ascending order) collection of
    /// unique item indices.
    ///
    /// Item indices start at 1 (index 0 is reserved). Empty options are
    /// silently ignored.
    ///
    /// # Panics
    ///
    /// * If an item index is encountered that is outside the bounds established
    ///   by the [`new`](Self::new) call (`1..=n_primary + n_secondary`).
    /// * If option items are not listed in ascending order or include duplicate
    ///   items
    pub fn add_option<'a, I: IntoIterator<Item = &'a usize>>(&mut self, option: I) -> &mut Self {
        let prev_spacer = self.dlx.v_links.len() - 1;
        let mut prev_item = 0; // 0 is not a valid item so we can use it as -INFINITY
        let mut current = prev_spacer;
        for &item in option {
            assert!(self.dlx.is_item(item));
            // The sort requirement may not be strictly necessary, erring on the side of caution
            // it also makes uniqueness easier to check
            assert!(
                item > prev_item,
                "option items must be unique and sorted ascending"
            );
            prev_item = item;

            let old_bottom = self.dlx.v_links[item].prev;
            self.dlx.v_links.push(DoubleIndexLink {
                prev: old_bottom,
                next: item,
            });
            self.dlx.data.push(item);
            current += 1;
            self.dlx.v_links[item].prev = current;
            self.dlx.v_links[old_bottom].next = current;

            self.dlx.data[item] += 1; // len
        }
        if current == prev_spacer {
            // empty iterator
            return self;
        }
        self.dlx.v_links[prev_spacer].next = current;
        // next spacer
        self.dlx.data.push(0); // spacer
        self.dlx.v_links.push(DoubleIndexLink {
            prev: prev_spacer + 1,
            next: 0,
        });
        self
    }

    /// Consumes the builder and returns the completed `Dlx`
    #[must_use]
    pub fn build(self) -> Dlx {
        self.dlx
    }
}

/// Provides a read-only view into the underlying `Dlx`
impl AsRef<Dlx> for DlxBuilder {
    fn as_ref(&self) -> &Dlx {
        &self.dlx
    }
}

/// Lower level "dancing links" data structure described by Knuth for solving
/// exact cover problems.
///
/// You probably don't need to work with this type directly unless you are
/// creating a custom implementation of [`ExactCoverProblem`]. See
/// [module](self) documentation for additional details on how to use the
/// default implementation.
///
/// `Dlx` supports optional secondary items which can be covered 0 or 1 times in
/// a valid solution, as the [`ExactCoverProblem::search`] algorithm needs no
/// modifications to support this extension.
///
/// Because this data structure involves rewriting linked list pointers on the
/// fly, calling its methods in the wrong order will likely silently corrupt the
/// links and yield garbage results. Read the documentation on its methods
/// carefully. Enable [`debug_assert!`] for some additional sanity checks when
/// debugging.

// ## Implementation Notes
//
// The data structure is conceptually an incidence matrix with columns corresponding to items and
// rows corresponding to options. Internally, the matrix is represented in a "sparse" condensed
// layout with non-zero entries converted to linked list nodes and empty / zero entries omitted. The
// first row consists of item column headers and is joined into two circularly linked lists
// (separate lists for primary and secondary items) with a separate header node for each list that
// does not correspond to any item. Each item column is joined as a circularly linked list using the
// item header as the list header node. "Spacer" pointer nodes are inserted around each option row
// to facilitate wrapping around when iterating over rows, but option rows are not otherwise linked
// because their layout is already contiguous. In other words, the sparse matrix format is
// conceptually a `Vec<DlxNode>`:
//
// ```
// enum DlxNode {
//     PrimaryItemsHeader {
//         h_link: DoubleIndexLink,
//     },
//     SecondaryItemsHeaderAndSpacer {
//         h_link: DoubleIndexLink,
//         wrap_links: DoubleIndexLink,
//     },
//     ItemColumnHeader {
//         h_link: DoubleIndexLink,
//         v_link: DoubleIndexLink,
//         len: usize,
//     },
//     OptionNode {
//         v_link: DoubleIndexLink,
//         item: std::num::NonZeroUsize,
//     },
//     Spacer {
//         wrap_links: DoubleIndexLink,
//     }
// }
// ```
//
// However, the actual sparse matrix is laid out in Structure of Arrays format instead of Array of
// Structures simply because that is how Knuth described it (it's also a bit more memory efficient).
//
// List pointers are implemented as integer indices into Vec's. Items occupy indices 1..=num_items,
// followed immediately by option and spacer nodes. The PrimaryItemsHeader is always index 0, while
// the SecondaryItemsHeaderAndSpacer is index `num_items + 1` and pulls double duty as the first
// spacer node for the first option row.
pub struct Dlx {
    /// horizontal links for item header row
    h_links: Vec<DoubleIndexLink>,
    /// vertical links for columns of item and option nodes; spacer nodes use this for wrap links
    v_links: Vec<DoubleIndexLink>,
    /// the meaning of data depends on the node type: for ordinary nodes it points to its item; for
    /// spacers, it is 0; for items it is the number of ordinary nodes currently in the column.
    ///
    /// todo: This overloading of meaning is a bit ugly and could use a rework to improve clarity
    data: Vec<usize>,
    /// stack of selected options
    selected_options: Vec<DlxOption>,
    /// item most recently selected by select_item and which does not have a selected option yet;
    /// don't need a full stack of items because items can be computed from the selected_options
    /// stack, except for the case when an item has been selected but an option has not
    current_item: Option<usize>,
}

impl Dlx {
    /// Covers `item` and sets it as the *current item*.
    ///
    /// Covering an item removes it from the set of outstanding uncovered items and marks all
    /// options that contain it as unavailabe for covering other options.
    ///
    /// The selected item must not already be covered, and there must not be a current item already
    /// set.
    ///
    /// Note that this method does not actually select an option, so in general, the next step
    /// after calling this method should be to [`select_option`](Self::select_option) or
    /// [`undo_item`](Self::undo_item) if no more options are available.
    pub fn select_item(&mut self, item: usize) {
        debug_assert!(self.current_item.is_none());
        debug_assert!(self.is_item(item));
        self.cover(item);
        self.current_item = Some(item);
    }

    /// Undoes [`select_item`](Self::select_item).
    ///
    /// # Panics
    ///
    /// If no current item is set (by `select_item`)
    pub fn undo_item(&mut self) {
        let item = self.current_item.expect("Current item must be set");
        self.uncover(item);
        self.current_item = None;
    }

    /// Returns the next available option that includes the *current item*. Returns the first
    /// available option if `prev` is `None`. Returns `None` if `prev` is the last available option
    /// for the item.
    ///
    /// # Panics
    ///
    /// If no current item is set (by `select_item`)
    #[must_use]
    pub fn next_option(&self, prev: Option<DlxOption>) -> Option<DlxOption> {
        let current_item = self
            .current_item
            .expect("Current item needs to be set to see available options");
        let prev = match prev {
            Some(DlxOption(option)) => {
                debug_assert!(self.is_option(option));
                debug_assert!(self.option_covers_current_item(option));
                option
            }
            None => current_item,
        };
        let next = self.v_links[prev].next;
        (!self.is_item(next)).then_some(DlxOption(next))
    }

    /// Adds option to the candidate solution and covers all uncovered items included in the option.
    ///
    /// This method must only be called when there is a *current item* set, and the option argument
    /// must include the current item. When called it unsets the current item.
    pub fn select_option(&mut self, option: DlxOption) {
        let option = option.0;
        debug_assert!(self.is_option(option));
        debug_assert!(self.option_covers_current_item(option));
        self.for_other_cw(option, |dlx, i| {
            dlx.cover(dlx.data[i]);
        });
        self.selected_options.push(DlxOption(option));
        self.current_item = None;
    }

    /// Undoes the most recent [`select_option`](Self::select_option) and returns the option that
    /// was deselected, or does nothing and returns `None` if there was nothing to undo.
    ///
    /// This method must not be called when there is a *current item*. It restores the previous
    /// current item if an option was undone.
    pub fn try_undo_option(&mut self) -> Option<DlxOption> {
        debug_assert!(self.current_item.is_none());
        self.selected_options.pop().map(|DlxOption(last_option)| {
            self.current_item = Some(self.data[last_option]);
            // It is important that this for_other go in the reverse order of select_option.
            self.for_other_ccw(last_option, |dlx, i| {
                dlx.uncover(dlx.data[i]);
            });
            DlxOption(last_option)
        })
    }

    /// Returns an iterator over the remaining uncovered primary items.
    pub fn primary_items(&self) -> impl Iterator<Item = usize> + '_ {
        LinkIterator::from_slice(&self.h_links, 0)
    }

    /// Returns the number of currently available options that include `item`.
    #[must_use]
    pub fn options_len(&self, item: usize) -> usize {
        debug_assert!(self.is_item(item));
        self.data[item]
    }

    /// Returns a slice of options in the currently selected solution, or `None` if not in a solved
    /// state. Use [`option_items`](Self::option_items) to get the items in the option.
    #[must_use]
    pub fn current_solution(&self) -> Option<&[DlxOption]> {
        (self.h_links[0].next == 0).then_some(&self.selected_options)
    }

    /// Returns a slice of item indices in the given option.
    #[must_use]
    pub fn option_items(&self, option: DlxOption) -> &[usize] {
        let option = option.0;
        debug_assert!(self.is_option(option));
        let mut spacer = option;
        while self.data[spacer] > 0 {
            spacer -= 1;
        }
        &self.data[spacer + 1..=self.v_links[spacer].next]
    }

    // private helper methods

    fn is_item(&self, i: usize) -> bool {
        (1..self.h_links.len() - 1).contains(&i)
    }

    fn is_option(&self, i: usize) -> bool {
        (self.h_links.len()..self.data.len()).contains(&i) && self.data[i] > 0
    }

    fn option_covers_current_item(&self, option: usize) -> bool {
        self.current_item == Some(self.data[option])
    }

    // Removes item from header list and [`hide`]s all of its options
    fn cover(&mut self, item: usize) {
        debug_assert!(self.is_item(item), "{item} must be an item");
        debug_assert!(
            !self.h_links.is_removed(item),
            "item {item} must not already be covered"
        );
        let mut node = self.v_links[item].next;
        while node != item {
            self.hide(node);
            node = self.v_links[node].next;
        }
        self.h_links.remove_links(item);
    }

    fn uncover(&mut self, item: usize) {
        debug_assert!(self.is_item(item), "{item} must be an item");
        debug_assert!(self.h_links.is_removed(item), "item {item} must be covered");
        let mut node = self.v_links[item].prev;
        while node != item {
            self.unhide(node);
            node = self.v_links[node].prev;
        }
        self.h_links.restore_links(item);
    }

    /// remove option from all items except the one corresponding to `node`
    fn hide(&mut self, node: usize) {
        self.for_other_cw(node, |dlx, i| {
            dlx.v_links.remove_links(i);
            let item = dlx.data[i];
            dlx.data[item] -= 1;
        });
    }

    fn unhide(&mut self, node: usize) {
        self.for_other_ccw(node, |dlx, i| {
            dlx.v_links.restore_links(i);
            let item = dlx.data[i];
            dlx.data[item] += 1;
        });
    }

    /// Calls a closure on every node in the same option/row as the `node` *except* for `node`
    /// itself. Nodes are traversed "clockwise", i.e. ascending order starting just after the
    /// argument node and wrapping around. The closure is passed `self` as the first argument.
    fn for_other_cw<F>(&mut self, node: usize, mut f: F)
    where
        F: FnMut(&mut Self, usize),
    {
        let mut i = node + 1;
        while i != node {
            if self.data[i] == 0 {
                // spacer node
                i = self.v_links[i].prev;
            } else {
                f(self, i);
                i += 1;
            }
        }
    }

    /// like [`for_other_cw`] but in the opposite direction, suitable for undoing
    fn for_other_ccw<F>(&mut self, node: usize, mut f: F)
    where
        F: FnMut(&mut Self, usize),
    {
        let mut i = node - 1;
        while i != node {
            if self.data[i] == 0 {
                // spacer node
                i = self.v_links[i].next;
            } else {
                f(self, i);
                i -= 1;
            }
        }
    }
}

/// An [`ExactCoverProblem`] implementation using the *minimum-remaining-values*
/// (MRV) heuristic, i.e. items with the fewest available options are selected
/// first.
///
/// See [Item Selection](ExactCoverProblem#item-selection) for more
/// details.
pub struct MrvExactCoverSearch {
    dlx: Dlx,
    option_cursor: Option<DlxOption>,
}

impl MrvExactCoverSearch {
    /// Creates a `MrvExactCoverSearch` from a pristine [`Dlx`] instance.
    ///
    /// # Panics
    ///
    /// If `dlx` has preselected items or options
    #[must_use]
    pub fn new(dlx: Dlx) -> Self {
        assert!(dlx.selected_options.is_empty() && dlx.current_item.is_none());
        Self {
            dlx,
            option_cursor: None,
        }
    }

    /// Returns the current solution (if one has been found) as a collection of
    /// "options", i.e. slices of items they cover.
    #[must_use]
    pub fn current_solution(&self) -> Option<impl IntoIterator<Item = &[usize]>> {
        self.dlx
            .current_solution()
            .map(|options| options.iter().map(|&option| self.dlx.option_items(option)))
    }
}

impl ExactCoverProblem for MrvExactCoverSearch {
    /// Chooses an item with the fewest available options.
    fn try_next_item(&mut self) -> bool {
        self.dlx
            .primary_items()
            .min_by_key(|&item| self.dlx.options_len(item))
            .map(|item| {
                self.dlx.select_item(item);
                self.option_cursor = None;
            })
            .is_some()
    }

    fn select_option_or_undo_item(&mut self) -> bool {
        if let Some(node) = self.dlx.next_option(self.option_cursor) {
            self.dlx.select_option(node);
            true
        } else {
            self.dlx.undo_item();
            false
        }
    }

    fn try_undo_option(&mut self) -> bool {
        self.dlx
            .try_undo_option()
            .map(|last_option| self.option_cursor = Some(last_option))
            .is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_items_backward(links: &[DoubleIndexLink], head: usize) -> usize {
        let mut n = 0;
        let mut i = head;
        while head != links[i].prev {
            n += 1;
            i = links[i].prev;
        }
        n
    }

    fn count_items_forward(links: &[DoubleIndexLink], head: usize) -> usize {
        let mut n = 0;
        let mut i = head;
        while head != links[i].next {
            n += 1;
            i = links[i].next;
        }
        n
    }

    #[test]
    fn test_dlx_new() {
        let x = DlxBuilder::new(4, 3).dlx;
        assert_eq!(count_items_forward(&x.h_links, 0), 4);
        assert_eq!(count_items_backward(&x.h_links, 0), 4);
        assert_eq!(count_items_forward(&x.h_links, x.h_links.len() - 1), 3);
        assert_eq!(count_items_backward(&x.h_links, x.h_links.len() - 1), 3);
        assert_eq!(x.data.len(), 9);
        assert_eq!(x.h_links.len(), 9);
        assert_eq!(x.v_links.len(), 9);
    }

    #[test]
    fn test_dlx_empty() {
        let x = DlxBuilder::new(0, 0).dlx;
        assert_eq!(count_items_forward(&x.h_links, 0), 0);
        assert_eq!(count_items_backward(&x.h_links, 0), 0);
        assert_eq!(count_items_forward(&x.h_links, 1), 0);
        assert_eq!(count_items_backward(&x.h_links, 1), 0);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_dlx_knuth_example() {
        let mut x = DlxBuilder::new(7, 0);
        x.add_option(&[3, 5]);
        x.add_option(&[1, 4, 7]);
        x.add_option(&[2, 3, 6]);
        x.add_option(&[1, 4, 6]);
        x.add_option(&[2, 7]);
        x.add_option(&[4, 5, 7]);
        let mut x = x.build();

        assert_eq!(x.data.len(), 31);
        assert_eq!(x.v_links.len(), 31);

        let collect_vlinks = |x: &Dlx| {
            x.v_links
                .iter()
                .zip(x.data.iter())
                .map(|(link, data)| (*data, link.prev, link.next))
                .collect::<Vec<_>>()
        };

        // Knuth uses negative data values in the spacer nodes but we just use 0
        let expected = [
            (0, 0, 0),
            (2, 20, 12),
            (2, 24, 16),
            (2, 17, 9),
            (3, 27, 13),
            (2, 28, 10),
            (2, 22, 18),
            (3, 29, 14),
            (0, 0, 10),
            (3, 3, 17),
            (5, 5, 28),
            (0, 9, 14),
            (1, 1, 20),
            (4, 4, 21),
            (7, 7, 25),
            (0, 12, 18),
            (2, 2, 24),
            (3, 9, 3),
            (6, 6, 22),
            (0, 16, 22),
            (1, 12, 1),
            (4, 13, 27),
            (6, 18, 6),
            (0, 20, 25),
            (2, 16, 2),
            (7, 14, 29),
            (0, 24, 29),
            (4, 21, 4),
            (5, 10, 5),
            (7, 25, 7),
            (0, 27, 0),
        ];
        assert_eq!(collect_vlinks(&x), expected);

        x.select_item(1);
        assert_eq!(x.primary_items().collect::<Vec<_>>(), [2, 3, 4, 5, 6, 7]);
        let expected = [
            (0, 0, 0),
            (2, 20, 12),
            (2, 24, 16),
            (2, 17, 9),
            (1, 27, 27),
            (2, 28, 10),
            (1, 18, 18),
            (2, 29, 25),
            (0, 0, 10),
            (3, 3, 17),
            (5, 5, 28),
            (0, 9, 14),
            (1, 1, 20),
            (4, 4, 21),
            (7, 7, 25),
            (0, 12, 18),
            (2, 2, 24),
            (3, 9, 3),
            (6, 6, 6),
            (0, 16, 22),
            (1, 12, 1),
            (4, 4, 27),
            (6, 18, 6),
            (0, 20, 25),
            (2, 16, 2),
            (7, 7, 29),
            (0, 24, 29),
            (4, 4, 4),
            (5, 10, 5),
            (7, 25, 7),
            (0, 27, 0),
        ];
        assert_eq!(collect_vlinks(&x), expected);

        x.select_option(DlxOption(12));
        assert_eq!(x.primary_items().collect::<Vec<_>>(), [2, 3, 5, 6]);
        assert_eq!(x.data[2..=6], [1, 2, 1, 1, 1]);

        assert_eq!(x.v_links[2].next, 16);
        assert_eq!(x.v_links[2].prev, 16);
        x.select_item(2);
        assert_eq!(x.current_item, Some(2));
        assert_eq!(x.next_option(None).unwrap().0, 16);
        assert!(x.next_option(Some(DlxOption(16))).is_none());
        x.select_option(DlxOption(16));
        assert_eq!(x.current_item, None);
        assert_eq!(count_items_forward(&x.h_links, 0), 1);
        assert_eq!(x.data[5], 0);
        assert_eq!(x.data[3], 1);
        assert_eq!(x.h_links[0].next, 5);
        assert_eq!(x.h_links[0].prev, 5);

        assert!(x.try_undo_option().is_some());
        assert_eq!(x.current_item, Some(2));
        x.undo_item();
        assert_eq!(x.current_item, None);
        assert_eq!(x.primary_items().collect::<Vec<_>>(), [2, 3, 5, 6]);
        assert_eq!(x.data[2..=6], [1, 2, 1, 1, 1]);
        assert_eq!(x.v_links[2].next, 16);
        assert_eq!(x.v_links[2].prev, 16);
    }

    #[test]
    fn solve_knuth_example() {
        let mut x = DlxBuilder::new(7, 0);
        x.add_option(&[3, 5]);
        x.add_option(&[1, 4, 7]);
        x.add_option(&[2, 3, 6]);
        x.add_option(&[1, 4, 6]);
        x.add_option(&[2, 7]);
        x.add_option(&[4, 5, 7]);

        let mut ec = MrvExactCoverSearch::new(x.build());

        // exactly one solution expected
        assert!(ec.search());
        let solution = ec
            .current_solution()
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(solution, [&[1, 4, 6][..], &[2, 7], &[3, 5],]);

        assert!(!ec.search());
        assert!(ec.current_solution().is_none());
    }
}
