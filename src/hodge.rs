//! # Hodge decomposition
//!
//! For the two-term cochain complex of a graph, `0 → C⁰ --δ--> C¹ → 0`, the
//! cochain spaces split orthogonally:
//!
//! ```text
//! C⁰ = ker(δ) ⊕ im(δ*)      C¹ = im(δ) ⊕ ker(δ*)
//! ```
//!
//! `ker(δ)` is H⁰ (the globally consistent assignments), `ker(δ*)` is H¹ (the
//! obstruction space). Everything this module returns is an **orthogonal
//! projection**, so the components of a cochain sum back to the input and are
//! mutually orthogonal.
//!
//! ## The layout is load-bearing
//!
//! The crate does not publish the index layout of C⁰ or C¹, so a user cannot
//! build a valid cochain vector without it. [`cochain_layout`] publishes it:
//! a 0-cochain's coordinates are laid out in vertex order, a 1-cochain's in
//! edge order, and each vertex/edge owns a contiguous block.
//!
//! ## ⚠ The exact-field trap
//!
//! `(δs)` is in `im(δ)` **by construction**, and `im(δ) ⊥ ker(δ*)`, so
//!
//! ```text
//! P_H¹(δs) ≡ 0   for every s.
//! ```
//!
//! A monitor that computes `δs` and projects it onto H¹ reports a healthy mesh
//! forever. That is correct mathematics, not a bug — but it means the C¹ path
//! only carries information about fields that are **not** exact coboundaries.
//!
//! ## Size bounds
//!
//! The Hodge path costs `Θ(E·n²·k³)` time — the same order as one `dim_h1`
//! call — because the dominant term is the RREF of `δ`.
//!
//! Memory is dominated by `im_delta`, which is `Θ(dim C¹ · rank(δ))` and so
//! about the size of the dense coboundary itself. Peak during construction is
//! roughly **2× the dense δ** (the RREF working copy plus the pivot-column
//! matrix).
//!
//! The one thing that can blow up is the **explicit H¹ basis**
//! ([`HodgeBasis::h1_representatives`]): `Θ(b1 · dim C¹)` = `Θ(E²k²)`, two
//! orders of magnitude larger than the dense coboundary. It is size-capped
//! rather than left to OOM — see that method for the budget.

use crate::{coboundary, Sheaf};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Which cochain space a decomposition lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CochainDegree {
    /// C⁰ — vertex assignments. Dimension = Σ_v dim(stalk(v)).
    C0,
    /// C¹ — edge fields. Dimension = Σ_e dim(codomain F_e).
    C1,
}

/// A cochain split into its orthogonal Hodge components.
///
/// A graph is a 1-complex, so `C² = 0`: a 0-cochain has no exact part and a
/// 1-cochain has no coexact part. The unused field is left empty rather than
/// omitted, so that a later higher-order complex can fill it in without a
/// breaking change.
#[derive(Clone, Debug)]
pub struct HodgeDecomposition {
    /// Which cochain space these components live in.
    pub degree: CochainDegree,
    /// The harmonic component: `ker(δ)` for C⁰, `ker(δ*)` for C¹.
    ///
    /// For C⁰ this is the **orthogonal projection onto `ker L`** — the globally
    /// consistent part of the input. It is a canonical representative of H⁰.
    /// For C¹ it is the canonical representative of H¹.
    pub harmonic: Vec<f64>,
    /// The exact component, in `im(δ)`. **Empty when `degree == C0`.**
    pub exact: Vec<f64>,
    /// The coexact component, in `im(δ*)`. **Empty when `degree == C1`.**
    pub coexact: Vec<f64>,
}

/// The four subspaces of the cochain complex, computed once and reusable.
///
/// Building this factorizes the coboundary. Reuse it across many cochains;
/// do not call the one-shot [`hodge_decompose_c0`] / [`hodge_decompose_c1`] in a
/// loop if you are decomposing more than a handful of cochains.
#[derive(Clone, Debug)]
pub struct HodgeBasis {
    /// `dim ker(δ)` — the dimension of H⁰.
    pub b0: usize,
    /// `dim ker(δ*)` — the dimension of H¹.
    pub b1: usize,
    /// `dim C⁰`.
    pub dim_c0: usize,
    /// `dim C¹`.
    pub dim_c1: usize,
    /// An **orthonormal** basis of `ker(δ) ⊂ C⁰`. `b0` vectors of length `dim_c0`.
    ///
    /// Small: `dim ker(δ)` is the number of independent global sections.
    pub ker_delta: Vec<Vec<f64>>,
    /// An **orthonormal** basis of `im(δ) ⊂ C¹`. `rank(δ)` vectors of length `dim_c1`.
    ///
    /// This is the large one — `Θ(dim_c1 · rank(δ))` storage. See the size
    /// bounds in the module documentation.
    pub im_delta: Vec<Vec<f64>>,
}

/// The index layout of the cochain spaces.
///
/// `coboundary()` places a 0-cochain's coordinates in vertex order and a
/// 1-cochain's in edge order, but does not publish the offsets. Use this to
/// build and to interpret cochain vectors.
#[derive(Clone, Debug)]
pub struct CochainLayout {
    /// Offset of each vertex's stalk block within C⁰. Length `num_vertices`.
    pub vertex_offsets: Vec<usize>,
    /// Offset of each edge's codomain block within C¹. Length `num_edges`.
    pub edge_offsets: Vec<usize>,
    /// `dim C⁰`.
    pub dim_c0: usize,
    /// `dim C¹`.
    pub dim_c1: usize,
}

/// The index layout of the cochain spaces of `sheaf`.
///
/// # Example
///
/// ```
/// use sheaf_cohomology::{cochain_layout, Sheaf};
///
/// let sheaf = Sheaf::constant(3, 2); // 3 vertices, stalk dim 2, 3 edges
/// let layout = cochain_layout(&sheaf);
///
/// assert_eq!(layout.vertex_offsets, vec![0, 2, 4]);
/// assert_eq!(layout.edge_offsets, vec![0, 2, 4]);
/// assert_eq!(layout.dim_c0, 6);
/// assert_eq!(layout.dim_c1, 6);
/// ```
pub fn cochain_layout(sheaf: &Sheaf) -> CochainLayout {
    // Vertex blocks: vertex `v` owns [vertex_offsets[v], vertex_offsets[v] + dim(stalk_v)).
    let mut vertex_offsets = Vec::with_capacity(sheaf.stalks.len());
    let mut dim_c0 = 0usize;
    for stalk in &sheaf.stalks {
        vertex_offsets.push(dim_c0);
        dim_c0 += stalk.0;
    }

    // Edge blocks: edge `e` owns [edge_offsets[e], edge_offsets[e] + dim(codomain F_e)).
    let mut edge_offsets = Vec::with_capacity(sheaf.edges.len());
    let mut dim_c1 = 0usize;
    for edge in &sheaf.edges {
        edge_offsets.push(dim_c1);
        dim_c1 += edge.matrix.len();
    }

    CochainLayout {
        vertex_offsets,
        edge_offsets,
        dim_c0,
        dim_c1,
    }
}

/// Factorize the coboundary once and return the Hodge subspaces.
///
/// # Cost
///
/// `Θ(E · n² · k³)` time and `Θ(dim_c1 · rank(δ))` memory — the same order as a
/// single `dim_h1` call, with roughly twice the memory of the dense coboundary.
///
/// # A note on `b0`
///
/// `b0` is computed as `dim C⁰ − rank(δ)`, **not** as `nullity(L)`. The
/// Laplacian is a Gram matrix (`L = δᵀδ`), so it squares `δ`'s condition
/// number, and the shipped `nullity(L)` path returns a wrong answer from
/// `n = 43` upward on a complete graph. The Hodge path never forms `L`, so it
/// stays correct where `dim_h0` does not.
pub fn hodge_basis(sheaf: &Sheaf) -> HodgeBasis {
    let delta = coboundary(sheaf);
    let dim_c0 = delta.cols;
    let dim_c1 = delta.rows;

    // δ is rendered into two private layouts: row-major for the Gauss-Jordan
    // reduction (which is inherently row-oriented) and column-major for the
    // column-oriented hot loops (Gram-Schmidt over the pivot columns).
    let mut work = vec![0.0f64; dim_c1 * dim_c0];
    let mut cols = ColMat::zero(dim_c1, dim_c0);
    for entry in &delta.entries {
        work[entry.row * dim_c0 + entry.col] += entry.val;
        cols.add(entry.row, entry.col, entry.val);
    }

    // (1) Factorize δ: rank, pivot columns, reduced row echelon form.
    let (rank, pivot_cols) = rref_with_pivots(&mut work, dim_c1, dim_c0);

    // (2) ker(δ) — the free columns of the RREF.
    let ker_delta = orthonormalise(null_space_from_rref(&work, dim_c0, &pivot_cols));
    drop(work); // the RREF working copy is no longer needed

    // (3) im(δ) — the pivot columns of δ span its column space.
    let pivot_columns: Vec<Vec<f64>> = pivot_cols.iter().map(|&c| cols.col(c).to_vec()).collect();
    let im_delta = orthonormalise(pivot_columns);

    HodgeBasis {
        b0: dim_c0.saturating_sub(rank),
        b1: dim_c1.saturating_sub(rank),
        dim_c0,
        dim_c1,
        ker_delta,
        im_delta,
    }
}

/// Decompose a 0-cochain into its harmonic and coexact components.
///
/// `s` must have length `dim C⁰`; the layout is given by [`cochain_layout`].
///
/// `harmonic` is the **orthogonal projection onto `ker L`** (equivalently
/// `ker(δ)`), `coexact = s − harmonic`, and `harmonic + coexact == s`.
/// `exact` is empty: a graph is a 1-complex, so `C² = 0`.
///
/// # Panics
///
/// If `s.len() != dim C⁰`.
pub fn hodge_decompose_c0(sheaf: &Sheaf, s: &[f64]) -> HodgeDecomposition {
    hodge_basis(sheaf).decompose_c0(s)
}

/// Decompose a 1-cochain into its exact and harmonic components.
///
/// `c` must have length `dim C¹`; the layout is given by [`cochain_layout`].
///
/// `exact` is the orthogonal projection onto `im(δ)` — the part some vertex
/// assignment explains. `harmonic` is the canonical representative of H¹:
/// the part **no** vertex assignment explains. `coexact` is empty.
///
/// # ⚠ The exact-field trap
///
/// `(δs)` is in `im(δ)` **by construction**, and `im(δ) ⊥ ker(δ*)`, so
///
/// ```text
/// P_H¹(δs) ≡ 0   for every s.
/// ```
///
/// Decomposing `δs` and reading off `harmonic` always reports a healthy mesh.
/// That is correct mathematics, not a bug — but it means this function only
/// carries information about fields that are **not** exact coboundaries.
///
/// # Panics
///
/// If `c.len() != dim C¹`.
pub fn hodge_decompose_c1(sheaf: &Sheaf, c: &[f64]) -> HodgeDecomposition {
    hodge_basis(sheaf).decompose_c1(c)
}

impl HodgeBasis {
    /// Decompose a 0-cochain, reusing the cached factorization.
    ///
    /// # Panics
    ///
    /// If `s.len() != dim C⁰`.
    pub fn decompose_c0(&self, s: &[f64]) -> HodgeDecomposition {
        assert_eq!(
            s.len(),
            self.dim_c0,
            "0-cochain must have length dim C⁰ = {}",
            self.dim_c0
        );
        let harmonic = self.project_harmonic_c0(s);
        let coexact: Vec<f64> = s.iter().zip(harmonic.iter()).map(|(a, b)| a - b).collect();
        HodgeDecomposition {
            degree: CochainDegree::C0,
            harmonic,
            exact: Vec::new(),
            coexact,
        }
    }

    /// Decompose a 1-cochain, reusing the cached factorization.
    ///
    /// # Panics
    ///
    /// If `c.len() != dim C¹`.
    pub fn decompose_c1(&self, c: &[f64]) -> HodgeDecomposition {
        assert_eq!(
            c.len(),
            self.dim_c1,
            "1-cochain must have length dim C¹ = {}",
            self.dim_c1
        );
        let exact = project_onto(&self.im_delta, c);
        let harmonic: Vec<f64> = c.iter().zip(exact.iter()).map(|(a, b)| a - b).collect();
        HodgeDecomposition {
            degree: CochainDegree::C1,
            harmonic,
            exact,
            coexact: Vec::new(),
        }
    }

    /// The orthogonal projection of a 0-cochain onto `ker(δ)` (equivalently `ker L`).
    ///
    /// # Panics
    ///
    /// If `s.len() != dim C⁰`.
    pub fn project_harmonic_c0(&self, s: &[f64]) -> Vec<f64> {
        assert_eq!(
            s.len(),
            self.dim_c0,
            "0-cochain must have length dim C⁰ = {}",
            self.dim_c0
        );
        project_onto(&self.ker_delta, s)
    }

    /// The orthogonal projection of a 1-cochain onto `ker(δ*)` — the H¹ component.
    ///
    /// # Panics
    ///
    /// If `c.len() != dim C¹`.
    pub fn project_harmonic_c1(&self, c: &[f64]) -> Vec<f64> {
        assert_eq!(
            c.len(),
            self.dim_c1,
            "1-cochain must have length dim C¹ = {}",
            self.dim_c1
        );
        // ker(δ*) = (im δ)⊥, so the H¹ component is what the projection onto
        // im(δ) does not explain.
        let exact = project_onto(&self.im_delta, c);
        c.iter().zip(exact.iter()).map(|(a, b)| a - b).collect()
    }

    /// A basis of `ker(δ*) ⊂ C¹`: the canonical representatives of H¹.
    ///
    /// These are the *edge supports* that name the offending cycles. They are
    /// computed from the RREF of `δᵀ` — an independent factorization of the
    /// adjoint, not the Gram matrix `L = δᵀδ` — so the vectors are exact in
    /// rank. The basis is **not** orthogonalised: it is meant to be read, and
    /// the vectors are sparse-ish in the sense that each one is pinned by a
    /// single free coordinate.
    ///
    /// # ⚠ Cost
    ///
    /// `Θ(b1 · dim_c1)` memory — for a complete graph that is `Θ(E²k²)`, which
    /// is **two orders of magnitude larger than the dense coboundary**. See the
    /// size bounds in the module documentation before calling this on a large
    /// sheaf.
    ///
    /// The allocation is capped at 32 MB (`b1 · dim_c1 ≤ 4·10⁶`); above the cap
    /// this method panics rather than exhausting memory. The H¹ *component* of a
    /// cochain is always available, at no extra cost, through
    /// [`HodgeBasis::project_harmonic_c1`] — only the explicit basis is capped.
    ///
    /// # Panics
    ///
    /// If the sheaf is not the one this basis was built from, or if the
    /// explicit basis would exceed the 32 MB budget.
    pub fn h1_representatives(&self, sheaf: &Sheaf) -> Vec<Vec<f64>> {
        assert_eq!(
            sheaf.dim_c1(),
            self.dim_c1,
            "sheaf does not match this HodgeBasis: dim C¹ is {} but the basis was built for {}",
            sheaf.dim_c1(),
            self.dim_c1
        );

        // Size cap: the explicit H¹ basis is Θ(b1 · dim C¹) = Θ(E²k²), two
        // orders of magnitude larger than the dense coboundary. Refuse rather
        // than OOM.
        let bytes = self
            .b1
            .saturating_mul(self.dim_c1)
            .saturating_mul(std::mem::size_of::<f64>());
        assert!(
            bytes <= H1_BASIS_MAX_BYTES,
            "the explicit H¹ basis would need {} bytes ({} vectors of length {}), \
             above the {} byte budget; use project_harmonic_c1 for the H¹ component instead",
            bytes,
            self.b1,
            self.dim_c1,
            H1_BASIS_MAX_BYTES
        );

        // δᵀ: dim C⁰ × dim C¹, row-major flat buffer.
        let delta = coboundary(sheaf);
        let dim_c0 = delta.cols;
        let dim_c1 = delta.rows;
        let mut dt = vec![0.0f64; dim_c0 * dim_c1];
        for entry in &delta.entries {
            dt[entry.col * dim_c1 + entry.row] += entry.val;
        }

        // ker(δ*) is the null space of δᵀ.
        let (rank_dt, pivot_cols) = rref_with_pivots(&mut dt, dim_c0, dim_c1);
        assert_eq!(
            rank_dt,
            self.dim_c0.saturating_sub(self.b0),
            "sheaf does not match this HodgeBasis: rank(δ) disagrees"
        );

        null_space_from_rref(&dt, dim_c1, &pivot_cols)
    }
}

// ---------------------------------------------------------------------------
// Private dense helpers (pure std, no dependencies)
// ---------------------------------------------------------------------------

/// Absolute pivot tolerance, matching the shipped `Matrix::rank` and
/// `Matrix::kernel_basis`.
const PIVOT_TOL: f64 = 1e-12;

/// Relative tolerance below which a column is treated as linearly dependent
/// during orthonormalisation.
const ORTHO_TOL: f64 = 1e-12;

/// Byte budget for the explicit H¹ basis (32 MB).
const H1_BASIS_MAX_BYTES: usize = 32 * 1024 * 1024;

/// A private, column-oriented dense matrix: element (row `r`, column `c`) is
/// at `data[c * rows + r]`.
///
/// The Hodge module's hot loops walk *columns* — Gram-Schmidt over the pivot
/// columns of `δ`, and the projector application. The shipped `Matrix` is
/// row-major with one heap allocation per row, so walking a column of it
/// strides across one allocation per row: a pointer chase per element. A
/// private flat, column-major buffer keeps those loops contiguous.
struct ColMat {
    rows: usize,
    data: Vec<f64>,
}

impl ColMat {
    fn zero(rows: usize, cols: usize) -> Self {
        ColMat {
            rows,
            data: vec![0.0; rows * cols],
        }
    }

    /// Column `c` as a contiguous slice.
    fn col(&self, c: usize) -> &[f64] {
        &self.data[c * self.rows..(c + 1) * self.rows]
    }

    fn add(&mut self, r: usize, c: usize, v: f64) {
        self.data[c * self.rows + r] += v;
    }
}

/// Gauss-Jordan reduction of a row-major flat buffer, in place.
///
/// This is the same elimination `Matrix::rank` performs, with the same
/// absolute pivot tolerance, but it also returns the pivot-column structure
/// that `Matrix::kernel_basis` computes internally and throws away.
///
/// Returns the rank and the pivot columns, in increasing order.
fn rref_with_pivots(work: &mut [f64], rows: usize, cols: usize) -> (usize, Vec<usize>) {
    let mut pivot_cols = Vec::new();
    let mut pr = 0usize;
    for col in 0..cols {
        if pr >= rows {
            break;
        }
        let mut pivot_row = None;
        for row in pr..rows {
            if work[row * cols + col].abs() > PIVOT_TOL {
                pivot_row = Some(row);
                break;
            }
        }
        if let Some(row) = pivot_row {
            if row != pr {
                for j in 0..cols {
                    work.swap(pr * cols + j, row * cols + j);
                }
            }
            let scale = work[pr * cols + col];
            for j in col..cols {
                work[pr * cols + j] /= scale;
            }
            for r in 0..rows {
                if r == pr {
                    continue;
                }
                let factor = work[r * cols + col];
                if factor.abs() > PIVOT_TOL {
                    for j in col..cols {
                        work[r * cols + j] -= factor * work[pr * cols + j];
                    }
                }
            }
            pivot_cols.push(col);
            pr += 1;
        }
    }
    (pr, pivot_cols)
}

/// The null space of a matrix whose reduced row echelon form is `work` (a
/// row-major flat buffer) with pivot columns `pivot_cols`.
///
/// One vector per free column, the same construction as `Matrix::kernel_basis`.
fn null_space_from_rref(work: &[f64], cols: usize, pivot_cols: &[usize]) -> Vec<Vec<f64>> {
    let mut is_pivot = vec![false; cols];
    for &pc in pivot_cols {
        is_pivot[pc] = true;
    }
    let mut basis = Vec::with_capacity(cols - pivot_cols.len());
    for fc in 0..cols {
        if is_pivot[fc] {
            continue;
        }
        let mut v = vec![0.0; cols];
        v[fc] = 1.0;
        for (ri, &pc) in pivot_cols.iter().enumerate() {
            v[pc] = -work[ri * cols + fc];
        }
        basis.push(v);
    }
    basis
}

/// Modified Gram-Schmidt with one reorthogonalisation pass.
///
/// The reorthogonalisation is not optional: plain Gram-Schmidt loses
/// orthogonality at `O(κ·ε)`, and these bases are used to build the
/// projectors, which are the whole point of the feature.
///
/// A vector whose remaining norm falls below `ORTHO_TOL` times its original
/// norm is linearly dependent and is dropped.
fn orthonormalise(basis: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(basis.len());
    for mut w in basis {
        let scale = norm(&w);
        if scale == 0.0 {
            continue;
        }
        // Pass 1: modified Gram-Schmidt. Pass 2: reorthogonalisation.
        for _ in 0..2 {
            for u in &out {
                let d = dot(u, &w);
                for (i, x) in w.iter_mut().enumerate() {
                    *x -= d * u[i];
                }
            }
        }
        let nrm = norm(&w);
        if nrm > ORTHO_TOL * scale {
            for x in w.iter_mut() {
                *x /= nrm;
            }
            out.push(w);
        }
    }
    out
}

/// Apply the orthogonal projector `P = U Uᵀ` to `v`, with `U` orthonormal.
///
/// `P` is never formed: that would be `Θ(dim C⁰²)` or `Θ(dim C¹²)`.
fn project_onto(basis: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for u in basis {
        let d = dot(u, v);
        if d == 0.0 {
            continue;
        }
        for (i, x) in out.iter_mut().enumerate() {
            *x += d * u[i];
        }
    }
    out
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{coboundary, dim_h1, Matrix, VertexStalk};

    fn identity(k: usize) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0; k]; k];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        m
    }

    /// A 0-cochain of the given length, with no obvious structure.
    fn probe(len: usize, seed: usize) -> Vec<f64> {
        (0..len)
            .map(|i| (((i * 37 + seed * 11) % 19) as f64 - 9.0) * 0.31)
            .collect()
    }

    /// The twisted triangle from the crate's own test suite: a flip edge forces
    /// one stalk coordinate to vanish, so H⁰ = 1 rather than 2.
    fn twisted_triangle() -> Sheaf {
        Sheaf {
            num_vertices: 3,
            stalks: vec![VertexStalk(2), VertexStalk(2), VertexStalk(2)],
            edges: vec![
                crate::EdgeRestriction {
                    i: 0,
                    j: 1,
                    matrix: identity(2),
                },
                crate::EdgeRestriction {
                    i: 1,
                    j: 2,
                    matrix: identity(2),
                },
                crate::EdgeRestriction {
                    i: 0,
                    j: 2,
                    matrix: vec![vec![1.0, 0.0], vec![0.0, -1.0]],
                },
            ],
        }
    }

    /// The projection sheaf: a 3-dimensional stalk restricted to a
    /// 2-dimensional one.
    fn projection_sheaf() -> Sheaf {
        Sheaf {
            num_vertices: 2,
            stalks: vec![VertexStalk(3), VertexStalk(2)],
            edges: vec![crate::EdgeRestriction {
                i: 0,
                j: 1,
                matrix: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            }],
        }
    }

    /// A zero restriction map: `(δs)_e = 0·s_i − s_j = −s_j`.
    fn zero_restriction_sheaf() -> Sheaf {
        Sheaf {
            num_vertices: 2,
            stalks: vec![VertexStalk(2), VertexStalk(2)],
            edges: vec![crate::EdgeRestriction {
                i: 0,
                j: 1,
                matrix: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            }],
        }
    }

    fn matvec(m: &Matrix, v: &[f64]) -> Vec<f64> {
        m.data
            .iter()
            .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
            .collect()
    }

    // -- T1: closed form ---------------------------------------------------

    #[test]
    fn test_closed_form_b0_and_b1_on_constant_sheaves() {
        // A constant sheaf on a connected graph has b0 = k and
        // b1 = (E − n + 1)·k.
        for n in 2..=6usize {
            for k in 1..=3usize {
                let sheaf = Sheaf::constant(n, k);
                let edges = sheaf.edges.len();
                let basis = hodge_basis(&sheaf);
                assert_eq!(basis.b0, k, "b0 should be k at n={n}, k={k}");
                assert_eq!(
                    basis.b1,
                    (edges + 1 - n) * k,
                    "b1 should be (E − n + 1)·k at n={n}, k={k}"
                );
            }
        }
    }

    // -- T2: agreement with the shipped dim_h1 ------------------------------

    #[test]
    fn test_b1_matches_shipped_dim_h1() {
        for sheaf in [
            Sheaf::constant(4, 2),
            Sheaf::constant_path(5, 3),
            twisted_triangle(),
            projection_sheaf(),
            zero_restriction_sheaf(),
        ] {
            let basis = hodge_basis(&sheaf);
            assert_eq!(basis.b1, dim_h1(&sheaf), "b1 must match dim_h1");
        }
    }

    // -- T3: b0 is the size of the ker(δ) basis ----------------------------

    #[test]
    fn test_b0_equals_size_of_ker_delta_basis() {
        for sheaf in [
            Sheaf::constant(4, 2),
            Sheaf::constant_path(5, 3),
            twisted_triangle(),
            projection_sheaf(),
        ] {
            let basis = hodge_basis(&sheaf);
            assert_eq!(basis.b0, basis.ker_delta.len(), "b0 == |ker(δ)|");
            assert_eq!(basis.dim_c0, sheaf.dim_c0());
            assert_eq!(basis.dim_c1, sheaf.dim_c1());
        }
    }

    // -- T5: the decomposition sums back ------------------------------------

    #[test]
    fn test_decomposition_sums_back() {
        let sheaf = Sheaf::constant(4, 2);
        let basis = hodge_basis(&sheaf);
        let s = probe(basis.dim_c0, 1);
        let d = basis.decompose_c0(&s);
        for ((h, co), x) in d.harmonic.iter().zip(d.coexact.iter()).zip(s.iter()) {
            assert!(
                (h + co - x).abs() < 1e-12,
                "harmonic + coexact must equal s"
            );
        }
        assert!(d.exact.is_empty(), "a 0-cochain has no exact part");
        assert_eq!(d.degree, CochainDegree::C0);

        let c = probe(basis.dim_c1, 2);
        let d1 = basis.decompose_c1(&c);
        for ((h, ex), x) in d1.harmonic.iter().zip(d1.exact.iter()).zip(c.iter()) {
            assert!(
                (h + ex - x).abs() < 1e-12,
                "harmonic + exact must equal c"
            );
        }
        assert!(d1.coexact.is_empty(), "a 1-cochain has no coexact part");
        assert_eq!(d1.degree, CochainDegree::C1);
    }

    // -- T6: the components are orthogonal ----------------------------------

    #[test]
    fn test_components_are_orthogonal() {
        let sheaf = Sheaf::constant(4, 2);
        let basis = hodge_basis(&sheaf);
        let s = probe(basis.dim_c0, 3);
        let d = basis.decompose_c0(&s);
        assert!(
            dot(&d.harmonic, &d.coexact).abs() < 1e-12,
            "harmonic ⟂ coexact"
        );

        let c = probe(basis.dim_c1, 4);
        let d1 = basis.decompose_c1(&c);
        assert!(dot(&d1.harmonic, &d1.exact).abs() < 1e-12, "harmonic ⟂ exact");
    }

    // -- T7: δ · harmonic == 0 ----------------------------------------------

    #[test]
    fn test_delta_times_harmonic_is_zero() {
        let sheaf = Sheaf::constant(4, 2);
        let basis = hodge_basis(&sheaf);
        let s = probe(basis.dim_c0, 5);
        let d = basis.decompose_c0(&s);
        let delta = coboundary(&sheaf).to_dense();
        let dh = matvec(&delta, &d.harmonic);
        assert!(norm(&dh) < 1e-12, "δ · harmonic must be zero");
    }

    // -- T8: coexact ∈ im(δ*) ------------------------------------------------

    #[test]
    fn test_coexact_is_orthogonal_to_ker_delta() {
        // im(δ*) = (ker δ)⊥, so the coexact part is orthogonal to ker(δ).
        let sheaf = Sheaf::constant(4, 2);
        let basis = hodge_basis(&sheaf);
        let s = probe(basis.dim_c0, 6);
        let d = basis.decompose_c0(&s);
        for u in &basis.ker_delta {
            assert!(
                dot(&d.coexact, u).abs() < 1e-12,
                "coexact must be orthogonal to ker(δ)"
            );
        }
    }

    // -- T9: δᵀ · harmonic == 0 (C¹) ----------------------------------------

    #[test]
    fn test_delta_star_times_harmonic_c1_is_zero() {
        let sheaf = Sheaf::constant(4, 2);
        let basis = hodge_basis(&sheaf);
        let c = probe(basis.dim_c1, 7);
        let d = basis.decompose_c1(&c);
        let delta = coboundary(&sheaf).to_dense();
        let dt = delta.transpose();
        let dt_h = matvec(&dt, &d.harmonic);
        assert!(norm(&dt_h) < 1e-12, "δᵀ · harmonic must be zero");
    }

    // -- T10: idempotence ---------------------------------------------------

    #[test]
    fn test_decomposition_is_idempotent() {
        let sheaf = Sheaf::constant(4, 2);
        let basis = hodge_basis(&sheaf);
        let s = probe(basis.dim_c0, 8);
        let first = basis.decompose_c0(&s);
        let second = basis.decompose_c0(&first.harmonic);
        for i in 0..first.harmonic.len() {
            assert!(
                (second.harmonic[i] - first.harmonic[i]).abs() < 1e-12,
                "projecting the harmonic part again must return it unchanged"
            );
        }
        assert!(
            second.coexact.iter().all(|v| v.abs() < 1e-12),
            "the harmonic part has no coexact part"
        );
    }

    // -- T11: the exact-field trap ------------------------------------------

    #[test]
    fn test_exact_field_trap() {
        // (δs) is in im(δ) by construction, so its H¹ component is zero.
        for sheaf in [
            Sheaf::constant(4, 2),
            Sheaf::constant_path(5, 3),
            twisted_triangle(),
        ] {
            let basis = hodge_basis(&sheaf);
            let s = probe(basis.dim_c0, 9);
            let delta = coboundary(&sheaf).to_dense();
            let ds = matvec(&delta, &s);
            let d = basis.decompose_c1(&ds);
            assert!(
                norm(&d.harmonic) < 1e-10,
                "P_H¹(δs) must be zero — the exact-field trap"
            );
        }
    }

    // -- T12: trees ----------------------------------------------------------

    #[test]
    fn test_tree_has_no_h1() {
        for (n, k) in [(2usize, 1usize), (4, 2), (6, 3)] {
            let sheaf = Sheaf::constant_path(n, k);
            let basis = hodge_basis(&sheaf);
            assert_eq!(basis.b1, 0, "H¹ must vanish on a tree");
            assert_eq!(basis.b0, k, "H⁰ = k on a tree with a constant sheaf");
        }
    }

    // -- T13: disconnected ---------------------------------------------------

    #[test]
    fn test_disconnected_b0_is_the_sum_of_the_components() {
        let sheaf = Sheaf {
            num_vertices: 3,
            stalks: vec![VertexStalk(1), VertexStalk(1), VertexStalk(1)],
            edges: vec![],
        };
        let basis = hodge_basis(&sheaf);
        assert_eq!(basis.b0, 3, "three isolated vertices: H⁰ = 3");
        assert_eq!(basis.b1, 0, "no edges: H¹ = 0");

        // Two components: an edge and an isolated vertex.
        let sheaf = Sheaf {
            num_vertices: 3,
            stalks: vec![VertexStalk(1), VertexStalk(1), VertexStalk(1)],
            edges: vec![crate::EdgeRestriction {
                i: 0,
                j: 1,
                matrix: vec![vec![1.0]],
            }],
        };
        assert_eq!(hodge_basis(&sheaf).b0, 2, "H⁰ = 1 + 1 over two components");
    }

    // -- T14: the twisted sheaf ---------------------------------------------

    #[test]
    fn test_twisted_sheaf_has_b0_one() {
        let basis = hodge_basis(&twisted_triangle());
        assert_eq!(basis.b0, 1, "the flip edge kills one coordinate");
    }

    // -- T15: the zero restriction map --------------------------------------

    #[test]
    fn test_zero_restriction_map_b0() {
        let basis = hodge_basis(&zero_restriction_sheaf());
        assert_eq!(basis.b0, 2, "a zero restriction map leaves the source free");
    }

    // -- T16: the projection sheaf ------------------------------------------

    #[test]
    fn test_projection_sheaf_decomposition() {
        let sheaf = projection_sheaf();
        let basis = hodge_basis(&sheaf);
        assert_eq!(basis.dim_c0, 5);
        assert_eq!(basis.dim_c1, 2);
        assert_eq!(basis.b0, 3, "the kernel of the projection is free");
        let s = probe(5, 10);
        let d = basis.decompose_c0(&s);
        assert!(norm(&d.coexact) > 0.0, "the projected-away direction is coexact");
    }

    // -- T19: the layout -----------------------------------------------------

    #[test]
    fn test_cochain_layout_offsets() {
        let sheaf = Sheaf::constant(3, 2);
        let layout = cochain_layout(&sheaf);
        assert_eq!(layout.vertex_offsets, vec![0, 2, 4]);
        assert_eq!(layout.edge_offsets, vec![0, 2, 4]);
        assert_eq!(layout.dim_c0, 6);
        assert_eq!(layout.dim_c1, 6);

        // Mixed stalk dimensions: the offsets follow the stalk sizes.
        let sheaf = Sheaf {
            num_vertices: 2,
            stalks: vec![VertexStalk(3), VertexStalk(2)],
            edges: vec![crate::EdgeRestriction {
                i: 0,
                j: 1,
                matrix: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            }],
        };
        let layout = cochain_layout(&sheaf);
        assert_eq!(layout.vertex_offsets, vec![0, 3]);
        assert_eq!(layout.edge_offsets, vec![0]);
        assert_eq!(layout.dim_c0, 5);
        assert_eq!(layout.dim_c1, 2);
    }

    #[test]
    fn test_cochain_layout_matches_the_coboundary() {
        // The published layout must be the one `coboundary()` actually uses.
        let sheaf = projection_sheaf();
        let layout = cochain_layout(&sheaf);
        let delta = coboundary(&sheaf);
        assert_eq!(delta.cols, layout.dim_c0);
        assert_eq!(delta.rows, layout.dim_c1);
    }

    // -- T20: wrong input length --------------------------------------------

    #[test]
    #[should_panic(expected = "0-cochain must have length dim C⁰")]
    fn test_wrong_c0_length_panics() {
        let sheaf = Sheaf::constant(3, 2);
        let _ = hodge_decompose_c0(&sheaf, &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "1-cochain must have length dim C¹")]
    fn test_wrong_c1_length_panics() {
        let sheaf = Sheaf::constant(3, 2);
        let _ = hodge_decompose_c1(&sheaf, &[1.0, 2.0]);
    }

    // -- the private column-major δ is the shipped dense δ ------------------

    #[test]
    fn test_private_column_oriented_delta_matches_to_dense() {
        let sheaf = Sheaf::constant(4, 2);
        let delta = coboundary(&sheaf);
        let dense = delta.to_dense();

        let mut cols = ColMat::zero(delta.rows, delta.cols);
        for entry in &delta.entries {
            cols.add(entry.row, entry.col, entry.val);
        }
        for c in 0..delta.cols {
            for r in 0..delta.rows {
                assert!(
                    (cols.col(c)[r] - dense.get(r, c)).abs() < 1e-15,
                    "the private column-major δ must equal coboundary().to_dense()"
                );
            }
        }
    }

    // -- the stored bases are orthonormal -----------------------------------

    #[test]
    fn test_stored_bases_are_orthonormal() {
        let sheaf = Sheaf::constant(4, 2);
        let basis = hodge_basis(&sheaf);
        for (name, vecs) in [("ker(δ)", &basis.ker_delta), ("im(δ)", &basis.im_delta)] {
            for (i, u) in vecs.iter().enumerate() {
                assert!(
                    (norm(u) - 1.0).abs() < 1e-12,
                    "{name} vector {i} must be a unit vector"
                );
                for (j, v) in vecs.iter().enumerate().skip(i + 1) {
                    assert!(
                        dot(u, v).abs() < 1e-12,
                        "{name} vectors {i} and {j} must be orthogonal"
                    );
                }
            }
        }
    }

    // -- the H¹ representatives span ker(δ*) --------------------------------

    #[test]
    fn test_h1_representatives_span_ker_delta_star() {
        let sheaf = Sheaf::constant(4, 2);
        let basis = hodge_basis(&sheaf);
        let reps = basis.h1_representatives(&sheaf);
        assert_eq!(reps.len(), basis.b1, "one representative per H¹ dimension");

        // Every representative must be in ker(δ*) ...
        let delta = coboundary(&sheaf).to_dense();
        let dt = delta.transpose();
        for u in &reps {
            assert!(
                norm(&matvec(&dt, u)) < 1e-10,
                "an H¹ representative must lie in ker(δ*)"
            );
        }

        // ... and they must span all of it. The representatives are a plain
        // (non-orthogonal) basis, so independence is checked by the dimension
        // of their span and by the projector they induce.
        let u = orthonormalise(reps.clone());
        assert_eq!(
            u.len(),
            basis.b1,
            "the H¹ representatives must be linearly independent"
        );
        let c = probe(basis.dim_c1, 11);
        let from_reps = project_onto(&u, &c);
        let from_complement = basis.project_harmonic_c1(&c);
        for i in 0..c.len() {
            assert!(
                (from_reps[i] - from_complement[i]).abs() < 1e-10,
                "the projector onto the H¹ representatives must agree with the H¹ component"
            );
        }
    }

    #[test]
    #[should_panic(expected = "above the")]
    fn test_h1_representatives_refuses_oversized_bases() {
        // n = 40, k = 4: b1 · dim C¹ = 2964 · 3120 elements = 74 MB, well
        // above the 32 MB budget.
        let sheaf = Sheaf::constant(40, 4);
        let basis = hodge_basis(&sheaf);
        let _ = basis.h1_representatives(&sheaf);
    }
}
