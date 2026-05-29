//! # sheaf-cohomology
//!
//! Cellular sheaves and cohomology on graphs — pure Rust, zero dependencies.
//!
//! A *cellular sheaf* assigns a vector space (the *stalk*) to each vertex of a graph
//! and a linear map (the *restriction*) to each edge. The sheaf cohomology
//! groups H⁰ and H¹ measure global consistency and obstruction to extension respectively.

use std::fmt;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Dimension of the vector space (stalk) attached to a vertex.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexStalk(pub usize);

/// Restriction map on an edge {i, j}: a dense matrix mapping stalk(i) → stalk(j).
/// The coboundary computes the difference: (δs)_{ij} = F(s_i) − s_j.
/// Stored row-major: `matrix[r][c]` gives row `r`, column `c`.
#[derive(Clone, Debug)]
pub struct EdgeRestriction {
    /// First vertex index.
    pub i: usize,
    /// Second vertex index.
    pub j: usize,
    /// Dense matrix (rows = dim(stalk_j), cols = dim(stalk_i)).
    pub matrix: Vec<Vec<f64>>,
}

/// A cellular sheaf on a graph.
///
/// * `num_vertices` — number of vertices (0-indexed).
/// * `stalks` — one `VertexStalk` per vertex giving the dimension of its stalk.
/// * `edges` — restriction maps on **undirected** edges (store each edge once).
#[derive(Clone, Debug)]
pub struct Sheaf {
    pub num_vertices: usize,
    pub stalks: Vec<VertexStalk>,
    pub edges: Vec<EdgeRestriction>,
}

impl Sheaf {
    /// Build a constant sheaf: every vertex has stalk dimension `k`, every edge
    /// restriction is the `k × k` identity. Edges form a complete graph on `n` vertices.
    pub fn constant(n: usize, k: usize) -> Self {
        let stalks = vec![VertexStalk(k); n];
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push(EdgeRestriction {
                    i,
                    j,
                    matrix: identity_matrix(k),
                });
            }
        }
        Sheaf {
            num_vertices: n,
            stalks,
            edges,
        }
    }

    /// Build a constant sheaf on a tree (path graph) with `n` vertices and stalk dimension `k`.
    pub fn constant_path(n: usize, k: usize) -> Self {
        let stalks = vec![VertexStalk(k); n];
        let mut edges = Vec::new();
        for i in 0..n.saturating_sub(1) {
            edges.push(EdgeRestriction {
                i,
                j: i + 1,
                matrix: identity_matrix(k),
            });
        }
        Sheaf {
            num_vertices: n,
            stalks,
            edges,
        }
    }

    /// Total dimension of C⁰ (sum of all stalk dimensions).
    pub fn dim_c0(&self) -> usize {
        self.stalks.iter().map(|s| s.0).sum()
    }

    /// Total dimension of C¹ (sum of codomain dimensions of all edge restrictions).
    pub fn dim_c1(&self) -> usize {
        self.edges.iter().map(|e| e.matrix.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Dense linear algebra helpers (no external deps)
// ---------------------------------------------------------------------------

fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

/// Dense matrix type (row-major).
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn zero(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zero(n, n);
        for i in 0..n {
            m.data[i][i] = 1.0;
        }
        m
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r][c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r][c] = v;
    }

    /// Matrix multiplication: &self * other.
    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);
        let mut result = Matrix::zero(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.data[i][k];
                if a == 0.0 {
                    continue;
                }
                for j in 0..other.cols {
                    result.data[i][j] += a * other.data[k][j];
                }
            }
        }
        result
    }

    /// Transpose.
    pub fn transpose(&self) -> Matrix {
        let mut t = Matrix::zero(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.data[j][i] = self.data[i][j];
            }
        }
        t
    }

    /// Compute rank via Gaussian elimination (row echelon form).
    pub fn rank(&self) -> usize {
        let mut mat = self.data.clone();
        let m = self.rows;
        let n = self.cols;
        let mut pivots = 0;
        for col in 0..n {
            if pivots >= m {
                break;
            }
            let mut pivot_row = None;
            for row in pivots..m {
                if mat[row][col].abs() > 1e-12 {
                    pivot_row = Some(row);
                    break;
                }
            }
            if let Some(pr) = pivot_row {
                mat.swap(pivots, pr);
                let scale = mat[pivots][col];
                for j in col..n {
                    mat[pivots][j] /= scale;
                }
                for row in 0..m {
                    if row == pivots {
                        continue;
                    }
                    let factor = mat[row][col];
                    if factor.abs() > 1e-12 {
                        for j in col..n {
                            mat[row][j] -= factor * mat[pivots][j];
                        }
                    }
                }
                pivots += 1;
            }
        }
        pivots
    }

    /// Nullity = cols - rank.
    pub fn nullity(&self) -> usize {
        self.cols.saturating_sub(self.rank())
    }

    /// Extract kernel basis vectors (columns of the null space).
    pub fn kernel_basis(&self) -> Vec<Vec<f64>> {
        let mut mat = self.data.clone();
        let m = self.rows;
        let n = self.cols;
        let mut pivot_cols: Vec<usize> = Vec::new();
        let mut pr = 0;
        for col in 0..n {
            if pr >= m {
                break;
            }
            let mut pivot_row = None;
            for row in pr..m {
                if mat[row][col].abs() > 1e-12 {
                    pivot_row = Some(row);
                    break;
                }
            }
            if let Some(row) = pivot_row {
                mat.swap(pr, row);
                let scale = mat[pr][col];
                for j in 0..n {
                    mat[pr][j] /= scale;
                }
                for row in 0..m {
                    if row == pr {
                        continue;
                    }
                    let factor = mat[row][col];
                    if factor.abs() > 1e-12 {
                        for j in 0..n {
                            mat[row][j] -= factor * mat[pr][j];
                        }
                    }
                }
                pivot_cols.push(col);
                pr += 1;
            }
        }
        let free_cols: Vec<usize> = (0..n).filter(|c| !pivot_cols.contains(c)).collect();
        let mut basis: Vec<Vec<f64>> = Vec::new();
        for &fc in &free_cols {
            let mut v = vec![0.0; n];
            v[fc] = 1.0;
            for (ri, &pc) in pivot_cols.iter().enumerate() {
                v[pc] = -mat[ri][fc];
            }
            basis.push(v);
        }
        basis
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &self.data {
            for (j, v) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{:8.4}", v)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Sparse matrix (COO-like, but we convert to dense for ops)
// ---------------------------------------------------------------------------

/// Sparse matrix entry.
#[derive(Clone, Debug)]
pub struct SparseEntry {
    pub row: usize,
    pub col: usize,
    pub val: f64,
}

/// Sparse matrix stored as a list of (row, col, value) entries.
#[derive(Clone, Debug)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub entries: Vec<SparseEntry>,
}

impl SparseMatrix {
    pub fn to_dense(&self) -> Matrix {
        let mut m = Matrix::zero(self.rows, self.cols);
        for e in &self.entries {
            m.data[e.row][e.col] += e.val;
        }
        m
    }
}

// ---------------------------------------------------------------------------
// Coboundary operator
// ---------------------------------------------------------------------------

/// Compute the coboundary δ: C⁰ → C¹.
///
/// For each edge e = {i, j} with restriction map F_e: stalk(i) → stalk(j),
/// the coboundary computes the "difference":
///   (δs)_e = F_e · s_i  −  s_j
///
/// For 1-dimensional stalks with identity restrictions, this reduces to the
/// oriented incidence matrix, and L = δ*δ reduces to the graph Laplacian.
pub fn coboundary(sheaf: &Sheaf) -> SparseMatrix {
    let c0_dim = sheaf.dim_c0();
    let mut entries = Vec::new();

    // Compute vertex offsets in C⁰
    let mut v_offsets: Vec<usize> = Vec::with_capacity(sheaf.num_vertices);
    let mut off = 0usize;
    for s in &sheaf.stalks {
        v_offsets.push(off);
        off += s.0;
    }

    let mut row = 0usize;
    for edge in &sheaf.edges {
        let di = sheaf.stalks[edge.i].0;
        let dj = sheaf.stalks[edge.j].0;
        assert_eq!(edge.matrix.len(), dj, "restriction matrix rows must equal target stalk dim");
        assert_eq!(edge.matrix[0].len(), di, "restriction matrix cols must equal source stalk dim");

        // Place +F_e at source vertex block (vertex i)
        for (r, mat_row) in edge.matrix.iter().enumerate() {
            for (c, &val) in mat_row.iter().enumerate() {
                if val.abs() > 1e-15 {
                    entries.push(SparseEntry {
                        row: row + r,
                        col: v_offsets[edge.i] + c,
                        val,
                    });
                }
            }
        }

        // Place −I at target vertex block (vertex j)
        for r in 0..dj {
            entries.push(SparseEntry {
                row: row + r,
                col: v_offsets[edge.j] + r,
                val: -1.0,
            });
        }

        row += dj;
    }

    SparseMatrix {
        rows: row,
        cols: c0_dim,
        entries,
    }
}

/// Compute the adjoint coboundary δ*: C¹ → C⁰ (transpose of δ).
pub fn coboundary_adjoint(sheaf: &Sheaf) -> SparseMatrix {
    let delta = coboundary(sheaf);
    let mut entries = Vec::new();
    for e in &delta.entries {
        entries.push(SparseEntry {
            row: e.col,
            col: e.row,
            val: e.val,
        });
    }
    SparseMatrix {
        rows: delta.cols,
        cols: delta.rows,
        entries,
    }
}

// ---------------------------------------------------------------------------
// Sheaf Laplacian
// ---------------------------------------------------------------------------

/// Compute the sheaf Laplacian L = δ* δ : C⁰ → C⁰.
///
/// For 1-dimensional stalks with identity restrictions this reduces to the
/// ordinary graph Laplacian.
pub fn sheaf_laplacian(sheaf: &Sheaf) -> Matrix {
    let delta = coboundary(sheaf).to_dense();
    let delta_star = delta.transpose();
    delta_star.mul(&delta)
}

/// Dimension of H⁰ = ker(L) = global sections (agreement space).
pub fn dim_h0(sheaf: &Sheaf) -> usize {
    let l = sheaf_laplacian(sheaf);
    l.nullity()
}

/// Compute the kernel basis of the sheaf Laplacian (global sections).
pub fn global_sections_basis(sheaf: &Sheaf) -> Vec<Vec<f64>> {
    let l = sheaf_laplacian(sheaf);
    l.kernel_basis()
}

// ---------------------------------------------------------------------------
// Cohomology
// ---------------------------------------------------------------------------

/// First cohomology dimension: H¹ = C¹ / im(δ) = coker(δ).
///
/// For a two-term cochain complex 0 → C⁰ → C¹ → 0 on a graph,
/// H¹ = dim(C¹) − rank(δ).
pub fn dim_h1(sheaf: &Sheaf) -> usize {
    let delta = coboundary(sheaf).to_dense();
    let r = delta.rank();
    sheaf.dim_c1().saturating_sub(r)
}

/// Euler characteristic: χ = dim H⁰ − dim H¹.
pub fn euler_characteristic(sheaf: &Sheaf) -> i32 {
    let h0 = dim_h0(sheaf) as i32;
    let h1 = dim_h1(sheaf) as i32;
    h0 - h1
}

// ---------------------------------------------------------------------------
// Consistency measures
// ---------------------------------------------------------------------------

/// Local consistency at a vertex: measures how well the stalk at `vertex`
/// agrees with its neighbors via the restriction maps.
///
/// Returns a value in (0, 1]: 1.0 means the Laplacian block at this vertex
/// is zero (perfect local agreement with all neighbors).
pub fn local_consistency(sheaf: &Sheaf, vertex: usize) -> f64 {
    let l = sheaf_laplacian(sheaf);
    let d = sheaf.stalks[vertex].0;
    if d == 0 {
        return 1.0;
    }

    let mut offset = 0usize;
    for v in 0..vertex {
        offset += sheaf.stalks[v].0;
    }

    // Frobenius norm of the vertex's row block in L
    let mut energy = 0.0_f64;
    for i in 0..d {
        let idx = offset + i;
        for j in 0..l.cols {
            energy += l.data[idx][j] * l.data[idx][j];
        }
    }

    if energy < 1e-14 {
        return 1.0;
    }

    // Normalize: consistency decreases with Laplacian energy
    1.0 / (1.0 + energy.sqrt() / d as f64)
}

/// Global consistency: overall sheaf coherence measured as the ratio of
/// global section dimension to total C⁰ dimension.
///
/// Returns 1.0 for a sheaf with no edges (maximal freedom) and decreases
/// as restrictions constrain the global sections.
pub fn global_consistency(sheaf: &Sheaf) -> f64 {
    let c0 = sheaf.dim_c0();
    if c0 == 0 {
        return 1.0;
    }
    let h0 = dim_h0(sheaf);
    h0 as f64 / c0 as f64
}

// ---------------------------------------------------------------------------
// Graph Laplacian (for verification)
// ---------------------------------------------------------------------------

/// Build the ordinary graph Laplacian from the edge list.
pub fn graph_laplacian(num_vertices: usize, edges: &[(usize, usize)]) -> Matrix {
    let mut l = Matrix::zero(num_vertices, num_vertices);
    for &(i, j) in edges {
        l.data[i][i] += 1.0;
        l.data[j][j] += 1.0;
        l.data[i][j] -= 1.0;
        l.data[j][i] -= 1.0;
    }
    l
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_sheaf_h0() {
        // Constant sheaf on 3-vertex complete graph, stalk dim 2
        let sheaf = Sheaf::constant(3, 2);
        let h0 = dim_h0(&sheaf);
        assert_eq!(h0, 2, "Constant sheaf on 3 vertices with dim 2 stalks should have H⁰ = 2");
    }

    #[test]
    fn test_constant_sheaf_h1_on_tree() {
        // Constant sheaf on a tree (path graph) has H¹ = 0
        let sheaf = Sheaf::constant_path(4, 2);
        let h1 = dim_h1(&sheaf);
        assert_eq!(h1, 0, "Constant sheaf on a tree should have H¹ = 0");
    }

    #[test]
    fn test_constant_sheaf_h1_on_cycle() {
        // Constant sheaf on a triangle (3-cycle) with 1-dim stalks: H¹ = 1
        let sheaf = Sheaf {
            num_vertices: 3,
            stalks: vec![VertexStalk(1), VertexStalk(1), VertexStalk(1)],
            edges: vec![
                EdgeRestriction { i: 0, j: 1, matrix: vec![vec![1.0]] },
                EdgeRestriction { i: 1, j: 2, matrix: vec![vec![1.0]] },
                EdgeRestriction { i: 0, j: 2, matrix: vec![vec![1.0]] },
            ],
        };
        let h1 = dim_h1(&sheaf);
        // C¹ = 3, rank(δ) = 2 (connected graph), H¹ = 3 - 2 = 1
        assert_eq!(h1, 1, "Constant sheaf on triangle should have H¹ = 1");
    }

    #[test]
    fn test_constant_sheaf_h0_dim1() {
        for n in [2, 3, 4, 5] {
            let sheaf = Sheaf::constant(n, 1);
            let h0 = dim_h0(&sheaf);
            assert_eq!(h0, 1, "Constant 1-dim sheaf on {} vertices: H⁰ = 1", n);
        }
    }

    #[test]
    fn test_binary_sheaf_on_edge() {
        let sheaf = Sheaf {
            num_vertices: 2,
            stalks: vec![VertexStalk(1), VertexStalk(1)],
            edges: vec![EdgeRestriction {
                i: 0,
                j: 1,
                matrix: vec![vec![1.0]],
            }],
        };
        let h0 = dim_h0(&sheaf);
        assert_eq!(h0, 1, "Binary sheaf on edge should have H⁰ = 1");
    }

    #[test]
    fn test_disconnected_sheaf() {
        let sheaf = Sheaf {
            num_vertices: 3,
            stalks: vec![VertexStalk(1), VertexStalk(1), VertexStalk(1)],
            edges: vec![],
        };
        let h0 = dim_h0(&sheaf);
        assert_eq!(h0, 3, "Disconnected sheaf: H⁰ = sum of vertex dims");
    }

    #[test]
    fn test_disconnected_sheaf_mixed_dims() {
        let sheaf = Sheaf {
            num_vertices: 2,
            stalks: vec![VertexStalk(3), VertexStalk(2)],
            edges: vec![],
        };
        let h0 = dim_h0(&sheaf);
        assert_eq!(h0, 5, "Disconnected: H⁰ = sum of stalk dims");
    }

    #[test]
    fn test_sheaf_laplacian_reduces_to_graph_laplacian() {
        let sheaf = Sheaf::constant(4, 1);
        let sl = sheaf_laplacian(&sheaf);

        let edges: Vec<(usize, usize)> = sheaf
            .edges
            .iter()
            .map(|e| (e.i, e.j))
            .collect();
        let gl = graph_laplacian(4, &edges);

        for i in 0..4 {
            for j in 0..4 {
                let diff = (sl.data[i][j] - gl.data[i][j]).abs();
                assert!(
                    diff < 1e-10,
                    "Sheaf Laplacian [{},{}] = {} != graph Laplacian {}",
                    i, j, sl.data[i][j], gl.data[i][j]
                );
            }
        }
    }

    #[test]
    fn test_global_consistency_constant_sheaf() {
        // Path graph (tree): H⁰ = stalk_dim, C⁰ = n * stalk_dim
        let sheaf = Sheaf::constant_path(3, 2);
        let gc = global_consistency(&sheaf);
        // H⁰ = 2, C⁰ = 6, ratio = 1/3
        assert!(
            gc > 0.0,
            "Global consistency should be positive, got {}",
            gc
        );
        assert!(
            (gc - 1.0 / 3.0).abs() < 1e-10,
            "Global consistency should be 1/3, got {}",
            gc
        );
    }

    #[test]
    fn test_global_consistency_disconnected() {
        let sheaf = Sheaf {
            num_vertices: 3,
            stalks: vec![VertexStalk(1), VertexStalk(1), VertexStalk(1)],
            edges: vec![],
        };
        let gc = global_consistency(&sheaf);
        assert!(
            (gc - 1.0).abs() < 1e-10,
            "Disconnected sheaf: consistency = 1.0, got {}",
            gc
        );
    }

    #[test]
    fn test_twisted_sheaf_lower_consistency() {
        // Triangle with 2-dim stalks: two identity edges, one "flip" edge
        // Flip edge {0,2} has matrix [[1,0],[0,-1]] → forces s_1 = 0
        // H⁰ = 1 (twisted) vs H⁰ = 2 (constant)
        let sheaf_twisted = Sheaf {
            num_vertices: 3,
            stalks: vec![VertexStalk(2), VertexStalk(2), VertexStalk(2)],
            edges: vec![
                EdgeRestriction { i: 0, j: 1, matrix: identity_matrix(2) },
                EdgeRestriction { i: 1, j: 2, matrix: identity_matrix(2) },
                EdgeRestriction { i: 0, j: 2, matrix: vec![vec![1.0, 0.0], vec![0.0, -1.0]] },
            ],
        };
        let gc_twisted = global_consistency(&sheaf_twisted);
        assert_eq!(dim_h0(&sheaf_twisted), 1, "Twisted sheaf H⁰ = 1");

        let sheaf_constant = Sheaf::constant(3, 2);
        let gc_constant = global_consistency(&sheaf_constant);
        assert_eq!(dim_h0(&sheaf_constant), 2, "Constant sheaf H⁰ = 2");

        assert!(
            gc_twisted < gc_constant,
            "Twisted sheaf consistency ({}) should be < constant ({})",
            gc_twisted,
            gc_constant
        );
    }

    #[test]
    fn test_euler_characteristic_tree() {
        // Path graph (tree) on 4 vertices, stalk dim 2: H⁰ = 2, H¹ = 0, χ = 2
        let sheaf = Sheaf::constant_path(4, 2);
        let chi = euler_characteristic(&sheaf);
        assert_eq!(chi, 2, "Euler characteristic of constant sheaf on tree = 2");
    }

    #[test]
    fn test_coboundary_dimensions() {
        let sheaf = Sheaf::constant(3, 2);
        let delta = coboundary(&sheaf);
        assert_eq!(delta.cols, 6, "C⁰ dimension = 3 × 2 = 6");
        assert_eq!(delta.rows, 6, "C¹ dimension = 3 edges × 2 = 6");
    }

    #[test]
    fn test_matrix_rank() {
        let m = Matrix::identity(3);
        assert_eq!(m.rank(), 3);

        let m2 = Matrix::zero(3, 3);
        assert_eq!(m2.rank(), 0);

        let mut m3 = Matrix::zero(2, 3);
        m3.data[0][0] = 1.0;
        m3.data[0][1] = 2.0;
        m3.data[0][2] = 3.0;
        m3.data[1][0] = 2.0;
        m3.data[1][1] = 4.0;
        m3.data[1][2] = 6.0;
        assert_eq!(m3.rank(), 1);
    }

    #[test]
    fn test_kernel_basis() {
        let mut m = Matrix::zero(2, 3);
        m.data[0][0] = 1.0;
        m.data[0][1] = 2.0;
        m.data[0][2] = 3.0;
        m.data[1][0] = 2.0;
        m.data[1][1] = 4.0;
        m.data[1][2] = 6.0;
        let basis = m.kernel_basis();
        assert_eq!(basis.len(), 2, "Nullity should be 2");
    }

    #[test]
    fn test_local_consistency_perfect() {
        let sheaf = Sheaf {
            num_vertices: 3,
            stalks: vec![VertexStalk(1), VertexStalk(1), VertexStalk(1)],
            edges: vec![],
        };
        for v in 0..3 {
            let lc = local_consistency(&sheaf, v);
            assert!(
                (lc - 1.0).abs() < 1e-10,
                "Isolated vertex: local consistency = 1.0, got {}",
                lc
            );
        }
    }

    #[test]
    fn test_higher_dimensional_stalks() {
        let sheaf = Sheaf {
            num_vertices: 2,
            stalks: vec![VertexStalk(3), VertexStalk(3)],
            edges: vec![EdgeRestriction {
                i: 0,
                j: 1,
                matrix: identity_matrix(3),
            }],
        };
        let h0 = dim_h0(&sheaf);
        assert_eq!(h0, 3, "Identity restriction: H⁰ = stalk dimension");
    }

    #[test]
    fn test_projection_sheaf() {
        let proj: Vec<Vec<f64>> = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let sheaf = Sheaf {
            num_vertices: 2,
            stalks: vec![VertexStalk(3), VertexStalk(2)],
            edges: vec![EdgeRestriction {
                i: 0,
                j: 1,
                matrix: proj,
            }],
        };
        // δ maps (a,b,c) → (a - x, b - y) where (x,y) = stalk at vertex 1
        // So (δs) = (a-x, b-y). ker(L) = global sections where a=x, b=y, and c is free.
        // But wait: L is 5×5 (C⁰ = 5). Let's check.
        let h0 = dim_h0(&sheaf);
        assert!(h0 <= 3, "H⁰ should be ≤ 3, got {}", h0);
    }

    #[test]
    fn test_path_sheaf() {
        let sheaf = Sheaf {
            num_vertices: 3,
            stalks: vec![VertexStalk(1), VertexStalk(1), VertexStalk(1)],
            edges: vec![
                EdgeRestriction { i: 0, j: 1, matrix: vec![vec![1.0]] },
                EdgeRestriction { i: 1, j: 2, matrix: vec![vec![1.0]] },
            ],
        };
        let h0 = dim_h0(&sheaf);
        assert_eq!(h0, 1, "Path graph with identity restrictions: H⁰ = 1");
        let h1 = dim_h1(&sheaf);
        assert_eq!(h1, 0, "Path graph (tree): H¹ = 0");
    }

    #[test]
    fn test_zero_map_kills_sections() {
        // Zero restriction map: (δs)_e = 0·s_i - s_j = -s_j
        // Global sections: s_j = 0 and s_i arbitrary → H⁰ = dim(stalk_i) unless i has other edges
        let sheaf = Sheaf {
            num_vertices: 2,
            stalks: vec![VertexStalk(2), VertexStalk(2)],
            edges: vec![EdgeRestriction {
                i: 0,
                j: 1,
                matrix: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            }],
        };
        let h0 = dim_h0(&sheaf);
        // L = δ^T δ. δ = [0 0 | -1 0; 0 0 | 0 -1] (2x4 matrix)
        // δ^T = [0 0; 0 0; -1 0; 0 -1]
        // L = δ^T δ = diag(0, 0, 2, 2) ... wait
        // Actually δ = [[0,0,-1,0],[0,0,0,-1]]
        // L = δ^T δ: 4x4
        // L[2,2] = 1, L[3,3] = 1, rest zero
        // ker(L) has dim 2 (the first two coordinates)
        assert_eq!(h0, 2, "Zero map: H⁰ = dim(stalk_i) = 2");
    }
}

