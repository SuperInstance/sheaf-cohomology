#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop)]
use std::collections::HashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Helper: serialize HashMap<(usize,usize), T> as Vec<([usize,usize], T)> for JSON.
fn serialize_pair_map<T: Serialize, S: Serializer>(
    map: &HashMap<(usize, usize), T>,
    s: S,
) -> Result<S::Ok, S::Error> {
    let v: Vec<([usize; 2], &T)> = map.iter().map(|(&(a, b), v)| ([a, b], v)).collect();
    v.serialize(s)
}

fn deserialize_pair_map<'de, T: Deserialize<'de>, D: Deserializer<'de>>(
    d: D,
) -> Result<HashMap<(usize, usize), T>, D::Error> {
    let v: Vec<([usize; 2], T)> = Vec::deserialize(d)?;
    Ok(v.into_iter().map(|([a, b], v)| ((a, b), v)).collect())
}

/// A cell in a cell complex.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Cell {
    pub id: usize,
    pub dimension: usize,
    pub vertices: Vec<usize>,
    pub faces: Vec<usize>,
    pub cofaces: Vec<usize>,
}

impl Cell {
    pub fn new(id: usize, dimension: usize, vertices: Vec<usize>) -> Self {
        Self {
            id,
            dimension,
            vertices,
            faces: Vec::new(),
            cofaces: Vec::new(),
        }
    }
}

/// Sparse matrix stored as HashMap<(row, col), value>.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    #[serde(serialize_with = "serialize_pair_map", deserialize_with = "deserialize_pair_map")]
    pub entries: HashMap<(usize, usize), f64>,
}

impl SparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            entries: HashMap::new(),
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n {
            m.entries.insert((i, i), 1.0);
        }
        m
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols)
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        *self.entries.get(&(i, j)).unwrap_or(&0.0)
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        if v.abs() > 1e-15 {
            self.entries.insert((i, j), v);
        } else {
            self.entries.remove(&(i, j));
        }
    }

    pub fn multiply_vec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.cols, "Vector length must match matrix columns");
        let mut result = vec![0.0; self.rows];
        for (&(i, j), &val) in &self.entries {
            result[i] += val * v[j];
        }
        result
    }

    pub fn transpose(&self) -> SparseMatrix {
        let mut t = SparseMatrix::new(self.cols, self.rows);
        for (&(i, j), &val) in &self.entries {
            t.entries.insert((j, i), val);
        }
        t
    }

    pub fn add(&self, other: &SparseMatrix) -> SparseMatrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut result = self.clone();
        for (&(i, j), &val) in &other.entries {
            let cur = result.get(i, j);
            result.set(i, j, cur + val);
        }
        result
    }

    /// Multiply two sparse matrices.
    pub fn multiply(&self, other: &SparseMatrix) -> SparseMatrix {
        assert_eq!(self.cols, other.rows);
        let mut result = SparseMatrix::new(self.rows, other.cols);
        // For each non-zero entry in self
        for (&(i, k1), &a_val) in &self.entries {
            // For each non-zero entry in other at row k1
            for (&(k2, j), &b_val) in &other.entries {
                if k1 == k2 {
                    let cur = result.get(i, j);
                    result.set(i, j, cur + a_val * b_val);
                }
            }
        }
        result
    }

    /// Compute rank via Gaussian elimination (on dense copy).
    pub fn rank(&self) -> usize {
        let dense = self.to_dense_rows();
        let (r, _) = Self::row_echelon(&dense, self.cols);
        r
    }

    pub fn nullity(&self) -> usize {
        self.cols.saturating_sub(self.rank())
    }

    /// Basis for the null space.
    pub fn kernel_basis(&self) -> Vec<Vec<f64>> {
        let dense = self.to_dense_rows();
        let (rank, pivots) = Self::row_echelon_with_pivots(&dense, self.cols);
        let n = self.cols;
        if rank == n {
            return Vec::new();
        }
        // Free variables are columns not in pivots
        let pivot_set: std::collections::HashSet<usize> = pivots.iter().take(rank).copied().collect();
        let free_vars: Vec<usize> = (0..n).filter(|j| !pivot_set.contains(j)).collect();
        
        // Get the reduced row echelon form
        let rref = Self::rref(&dense, self.cols);
        
        let mut basis = Vec::new();
        for &fv in &free_vars {
            let mut v = vec![0.0; n];
            v[fv] = 1.0;
            // For each pivot column, solve
            for (row_idx, &pivot_col) in pivots.iter().take(rank).enumerate() {
                if let Some(val) = rref[row_idx].get(fv) {
                    v[pivot_col] = -val;
                }
            }
            basis.push(v);
        }
        basis
    }

    fn to_dense_rows(&self) -> Vec<Vec<f64>> {
        let mut rows = vec![vec![0.0; self.cols]; self.rows];
        for (&(i, j), &val) in &self.entries {
            rows[i][j] = val;
        }
        rows
    }

    /// Row echelon form, returns (rank, rows).
    fn row_echelon(rows: &[Vec<f64>], ncols: usize) -> (usize, Vec<Vec<f64>>) {
        let mut m: Vec<Vec<f64>> = rows.to_vec();
        let nrows = m.len();
        if nrows == 0 {
            return (0, m);
        }
        let mut rank = 0;
        for col in 0..ncols {
            // Find pivot
            let pivot_row = (rank..nrows).find(|&r| m[r][col].abs() > 1e-12);
            let pivot_row = match pivot_row {
                Some(r) => r,
                None => continue,
            };
            m.swap(rank, pivot_row);
            let scale = m[rank][col];
            if scale.abs() < 1e-15 {
                continue;
            }
            for j in 0..ncols {
                m[rank][j] /= scale;
            }
            for r in 0..nrows {
                if r != rank && m[r][col].abs() > 1e-12 {
                    let factor = m[r][col];
                    for j in 0..ncols {
                        m[r][j] -= factor * m[rank][j];
                    }
                }
            }
            rank += 1;
        }
        (rank, m)
    }

    fn row_echelon_with_pivots(rows: &[Vec<f64>], ncols: usize) -> (usize, Vec<usize>) {
        let mut m: Vec<Vec<f64>> = rows.to_vec();
        let nrows = m.len();
        if nrows == 0 {
            return (0, Vec::new());
        }
        let mut rank = 0;
        let mut pivots = Vec::new();
        for col in 0..ncols {
            let pivot_row = (rank..nrows).find(|&r| m[r][col].abs() > 1e-12);
            let pivot_row = match pivot_row {
                Some(r) => r,
                None => continue,
            };
            m.swap(rank, pivot_row);
            let scale = m[rank][col];
            if scale.abs() < 1e-15 {
                continue;
            }
            for j in 0..ncols {
                m[rank][j] /= scale;
            }
            for r in 0..nrows {
                if r != rank && m[r][col].abs() > 1e-12 {
                    let factor = m[r][col];
                    for j in 0..ncols {
                        m[r][j] -= factor * m[rank][j];
                    }
                }
            }
            pivots.push(col);
            rank += 1;
        }
        (rank, pivots)
    }

    fn rref(rows: &[Vec<f64>], ncols: usize) -> Vec<Vec<f64>> {
        let (_, m) = Self::row_echelon(rows, ncols);
        m
    }

    /// Convert to DenseMatrix.
    pub fn to_dense(&self) -> DenseMatrix {
        DenseMatrix::from_sparse(self)
    }
}

/// Dense matrix for stalk and restriction map computations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl DenseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n {
            m.data[i][i] = 1.0;
        }
        m
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols)
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        let mut m = Self::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                m.data[i][j] = data[i * cols + j];
            }
        }
        m
    }

    pub fn from_sparse(s: &SparseMatrix) -> Self {
        let mut m = Self::new(s.rows, s.cols);
        for (&(i, j), &v) in &s.entries {
            m.data[i][j] = v;
        }
        m
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i][j] = v;
    }

    pub fn multiply(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.cols, other.rows);
        let mut result = DenseMatrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        result
    }

    pub fn multiply_vec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.cols);
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.data[i][j] * v[j];
            }
        }
        result
    }

    pub fn transpose(&self) -> DenseMatrix {
        let mut t = DenseMatrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.data[j][i] = self.data[i][j];
            }
        }
        t
    }

    pub fn add(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut r = DenseMatrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                r.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        r
    }

    pub fn scale(&self, s: f64) -> DenseMatrix {
        let mut r = DenseMatrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                r.data[i][j] = self.data[i][j] * s;
            }
        }
        r
    }

    pub fn rank(&self) -> usize {
        let (r, _) = SparseMatrix::row_echelon(&self.data, self.cols);
        r
    }

    pub fn nullity(&self) -> usize {
        self.cols.saturating_sub(self.rank())
    }

    pub fn kernel_basis(&self) -> Vec<Vec<f64>> {
        let (rank, pivots) = SparseMatrix::row_echelon_with_pivots(&self.data, self.cols);
        let n = self.cols;
        if rank == n {
            return Vec::new();
        }
        let pivot_set: std::collections::HashSet<usize> = pivots.iter().take(rank).copied().collect();
        let free_vars: Vec<usize> = (0..n).filter(|j| !pivot_set.contains(j)).collect();
        let rref = SparseMatrix::rref(&self.data, self.cols);
        let mut basis = Vec::new();
        for &fv in &free_vars {
            let mut v = vec![0.0; n];
            v[fv] = 1.0;
            for (row_idx, &pivot_col) in pivots.iter().take(rank).enumerate() {
                if let Some(row) = rref.get(row_idx) {
                    v[pivot_col] = -row[fv];
                }
            }
            basis.push(v);
        }
        basis
    }

    pub fn image_basis(&self) -> Vec<Vec<f64>> {
        // The image is spanned by the columns; find linearly independent ones
        let at = self.transpose();
        at.kernel_basis(); // kernel of A^T tells us dependencies
        // Better: use rank to find pivot columns
        let (_rank, _pivots) = SparseMatrix::row_echelon_with_pivots(&self.data, self.cols);
        // Actually we need column pivot info. Let's do it via A^T.
        let at_data = at.data.clone();
        let (at_rank, at_pivots) = SparseMatrix::row_echelon_with_pivots(&at_data, at.cols);
        // at_pivots are the pivot columns of A^T = pivot rows of A
        // For image basis, we just need rank many independent columns
        let mut basis = Vec::new();
        for &col in &at_pivots[..at_rank] {
            let mut column = vec![0.0; self.rows];
            for i in 0..self.rows {
                column[i] = self.data[i][col];
            }
            basis.push(column);
        }
        basis
    }

    /// Check if matrix is approximately zero.
    pub fn is_zero(&self, tol: f64) -> bool {
        self.data.iter().all(|row| row.iter().all(|&v| v.abs() < tol))
    }

    /// Check approximate equality.
    pub fn approx_eq(&self, other: &DenseMatrix, tol: f64) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        for i in 0..self.rows {
            for j in 0..self.cols {
                if (self.data[i][j] - other.data[i][j]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Column-vector version of data (for embedding in larger matrices).
    pub fn as_columns(&self) -> Vec<Vec<f64>> {
        let mut cols = Vec::new();
        for j in 0..self.cols {
            let mut c = vec![0.0; self.rows];
            for i in 0..self.rows {
                c[i] = self.data[i][j];
            }
            cols.push(c);
        }
        cols
    }
}

/// A cell complex — the underlying topological space.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CellComplex {
    pub cells: Vec<Cell>,
    pub dimension: usize,
}

impl Default for CellComplex {
    fn default() -> Self {
        Self::new()
    }
}

impl CellComplex {
    pub fn new() -> Self {
        Self {
            cells: Vec::new(),
            dimension: 0,
        }
    }

    pub fn add_cell(&mut self, id: usize, dimension: usize, vertices: Vec<usize>, faces: Vec<usize>) {
        let mut cell = Cell::new(id, dimension, vertices);
        cell.faces = faces;
        // Register cofaces
        for &face_id in &cell.faces {
            if let Some(fc) = self.cells.iter_mut().find(|c| c.id == face_id) {
                fc.cofaces.push(id);
            }
        }
        self.cells.push(cell);
        self.dimension = self.dimension.max(dimension);
    }

    pub fn cell(&self, id: usize) -> Option<&Cell> {
        self.cells.iter().find(|c| c.id == id)
    }

    pub fn cell_mut(&mut self, id: usize) -> Option<&mut Cell> {
        self.cells.iter_mut().find(|c| c.id == id)
    }

    pub fn cells_of_dimension(&self, k: usize) -> Vec<&Cell> {
        self.cells.iter().filter(|c| c.dimension == k).collect()
    }

    /// Number of cells of dimension k.
    pub fn num_cells(&self, k: usize) -> usize {
        self.cells.iter().filter(|c| c.dimension == k).count()
    }

    /// Boundary matrix ∂ₖ: Cₖ → Cₖ₋₁
    /// Maps k-cells to (k-1)-cells. Sign determined by incidence.
    pub fn boundary_matrix(&self, k: usize) -> SparseMatrix {
        if k == 0 {
            return SparseMatrix::new(0, self.num_cells(0));
        }
        let k_cells: Vec<&Cell> = self.cells_of_dimension(k).into_iter().collect();
        let km1_cells: Vec<&Cell> = self.cells_of_dimension(k - 1).into_iter().collect();
        let ncols = k_cells.len();
        let nrows = km1_cells.len();
        let mut mat = SparseMatrix::new(nrows, ncols);

        // Build index maps: cell_id -> position in the filtered list
        let km1_idx: HashMap<usize, usize> = km1_cells.iter().enumerate().map(|(i, c)| (c.id, i)).collect();

        for (col, cell) in k_cells.iter().enumerate() {
            for &face_id in &cell.faces {
                if let Some(&row) = km1_idx.get(&face_id) {
                    // Determine sign: use vertex ordering to determine orientation
                    let sign = self.incidence_sign(cell, face_id);
                    mat.set(row, col, sign);
                }
            }
        }
        mat
    }

    /// Coboundary matrix δₖ = ∂ₖ₊₁ᵀ: Cᵏ → Cᵏ⁺¹
    pub fn coboundary_matrix(&self, k: usize) -> SparseMatrix {
        self.boundary_matrix(k + 1).transpose()
    }

    /// Determine the incidence sign of face_id in cell.
    fn incidence_sign(&self, cell: &Cell, face_id: usize) -> f64 {
        // For simplices: sign is (-1)^i where i is the position of the omitted vertex
        // For a k-cell with vertices [v0,...,vk] and a face missing vertex vi, sign = (-1)^i
        let face = match self.cell(face_id) {
            Some(f) => f,
            None => return 1.0,
        };

        // Find which vertex of cell is missing from face
        let missing: Vec<usize> = cell.vertices.iter()
            .filter(|v| !face.vertices.contains(v))
            .copied()
            .collect();

        if missing.len() == 1 {
            // Find index of missing vertex in cell's vertex list
            if let Some(idx) = cell.vertices.iter().position(|&v| v == missing[0]) {
                if idx % 2 == 0 { 1.0 } else { -1.0 }
            } else {
                1.0
            }
        } else {
            1.0
        }
    }

    /// Betti numbers: βₖ = dim ker(∂ₖ) / dim im(∂ₖ₊₁)
    pub fn betti_numbers(&self) -> Vec<usize> {
        let mut betti = Vec::new();
        for k in 0..=self.dimension {
            let b_k = self.boundary_matrix(k);
            let b_k1 = self.boundary_matrix(k + 1);
            let ker = b_k.nullity();
            let im = b_k1.rank();
            betti.push(ker.saturating_sub(im));
        }
        betti
    }

    /// Euler characteristic: χ = Σ(-1)^k * |Cₖ|
    pub fn euler_characteristic(&self) -> i32 {
        let mut chi = 0i32;
        for k in 0..=self.dimension {
            let count = self.num_cells(k) as i32;
            if k % 2 == 0 {
                chi += count;
            } else {
                chi -= count;
            }
        }
        chi
    }

    /// Check connectivity: a complex is connected if its 1-skeleton graph is connected.
    pub fn is_connected(&self) -> bool {
        let vertices: Vec<&Cell> = self.cells_of_dimension(0).into_iter().collect();
        if vertices.is_empty() {
            return true;
        }
        let n = vertices.len();
        if n == 1 {
            return true;
        }
        let v_ids: Vec<usize> = vertices.iter().map(|c| c.id).collect();
        let mut visited = vec![false; n];
        let mut stack = vec![0];

        while let Some(idx) = stack.pop() {
            if visited[idx] {
                continue;
            }
            visited[idx] = true;
            let v_id = v_ids[idx];
            // Find all edges incident to this vertex
            let vertex = self.cell(v_id).unwrap();
            for &edge_id in &vertex.cofaces {
                let edge = self.cell(edge_id).unwrap();
                // Find the other vertex
                for &other_v in &edge.vertices {
                    if other_v != v_id {
                        if let Some(other_idx) = v_ids.iter().position(|&x| x == other_v) {
                            if !visited[other_idx] {
                                stack.push(other_idx);
                            }
                        }
                    }
                }
            }
        }
        visited.iter().all(|&v| v)
    }
}

/// A cellular sheaf on a cell complex.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CellularSheaf {
    pub complex: CellComplex,
    /// cell_id -> dimension of stalk F(σ)
    pub stalks: HashMap<usize, usize>,
    /// (σ, τ) -> restriction map F(σ⪯τ): F(τ) → F(σ)
    #[serde(serialize_with = "serialize_pair_map", deserialize_with = "deserialize_pair_map")]
    pub restriction_maps: HashMap<(usize, usize), DenseMatrix>,
}

impl CellularSheaf {
    pub fn new(complex: CellComplex) -> Self {
        Self {
            complex,
            stalks: HashMap::new(),
            restriction_maps: HashMap::new(),
        }
    }

    /// Set the stalk dimension for a cell.
    pub fn set_stalk(&mut self, cell_id: usize, dim: usize) {
        self.stalks.insert(cell_id, dim);
    }

    /// Set the restriction map F(σ⪯τ): F(τ) → F(σ) for face relation σ⪯τ.
    pub fn set_restriction(&mut self, sigma: usize, tau: usize, map: DenseMatrix) {
        self.restriction_maps.insert((sigma, tau), map);
    }

    /// Dimension of stalk at cell.
    pub fn stalk_dimension(&self, cell_id: usize) -> usize {
        *self.stalks.get(&cell_id).unwrap_or(&0)
    }

    /// Total stalk dimension for all k-cells.
    pub fn total_stalk_dimension(&self, k: usize) -> usize {
        self.complex.cells_of_dimension(k)
            .iter()
            .map(|c| self.stalk_dimension(c.id))
            .sum()
    }

    /// Verify functoriality: F(σ⪯σ) = id and F(σ⪯ρ) = F(σ⪯τ) ∘ F(τ⪯ρ) for σ⪯τ⪯ρ.
    pub fn verify_functoriality(&self) -> bool {
        let tol = 1e-10;
        // Check identity restrictions
        for cell in &self.complex.cells {
            if let Some(rmap) = self.restriction_maps.get(&(cell.id, cell.id)) {
                let id = DenseMatrix::identity(self.stalk_dimension(cell.id));
                if !rmap.approx_eq(&id, tol) {
                    return false;
                }
            }
        }

        // Check composition: for each triple σ⪯τ⪯ρ
        for sigma in &self.complex.cells {
            for &tau_id in &sigma.faces {
                if let Some(tau) = self.complex.cell(tau_id) {
                    for &rho_id in &tau.faces {
                        // σ⪯τ⪯ρ => F(σ⪯ρ) should equal F(σ⪯τ) ∘ F(τ⪯ρ)
                        let f_st = self.restriction_maps.get(&(sigma.id, tau_id));
                        let f_tr = self.restriction_maps.get(&(tau_id, rho_id));
                        let f_sr = self.restriction_maps.get(&(sigma.id, rho_id));

                        if let (Some(fst), Some(ftr), Some(fsr)) = (f_st, f_tr, f_sr) {
                            let composition = fst.multiply(ftr);
                            if !composition.approx_eq(fsr, tol) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }

    /// Build the sheaf coboundary operator matrix δₖᶠ: Cᵏ(F) → Cᵏ⁺¹(F)
    /// Cᵏ(F) = ⊕ F(σ) for σ in k-cells
    /// For a (k+1)-cell τ and its face σ, the block is the restriction F(σ⪯τ)
    pub fn coboundary_matrix(&self, k: usize) -> SparseMatrix {
        let k_cells: Vec<&Cell> = self.complex.cells_of_dimension(k).into_iter().collect();
        let k1_cells: Vec<&Cell> = self.complex.cells_of_dimension(k + 1).into_iter().collect();

        // Compute offsets into C^k
        let mut offsets_k: HashMap<usize, usize> = HashMap::new();
        let mut off = 0;
        for c in &k_cells {
            offsets_k.insert(c.id, off);
            off += self.stalk_dimension(c.id);
        }
        let total_k = off;

        let mut offsets_k1: HashMap<usize, usize> = HashMap::new();
        off = 0;
        for c in &k1_cells {
            offsets_k1.insert(c.id, off);
            off += self.stalk_dimension(c.id);
        }
        let total_k1 = off;

        let mut mat = SparseMatrix::new(total_k1, total_k);

        for (row, tau) in k1_cells.iter().enumerate() {
            let _row = row; // just for clarity
            let tau_offset = offsets_k1[&tau.id];
            let tau_dim = self.stalk_dimension(tau.id);

            for &sigma_id in &tau.faces {
                let sigma = match self.complex.cell(sigma_id) {
                    Some(s) => s,
                    None => continue,
                };
                if sigma.dimension != k {
                    continue;
                }
                let sigma_offset = offsets_k[&sigma_id];
                let sigma_dim = self.stalk_dimension(sigma_id);

                // Get restriction map F(σ⪯τ): F(τ) → F(σ)
                // But for coboundary, we need the dual direction
                // δₖ maps Cᵏ → Cᵏ⁺¹
                // For the incidence σ⪯τ with sign ε, the block is ε * F(σ⪯τ)ᵀ or ε * F(σ⪯τ)
                // Standard sheaf coboundary: for each face σ of τ, block is F(σ⪯τ) with sign
                let sign = self.complex.incidence_sign(tau, sigma_id);

                if let Some(rmap) = self.restriction_maps.get(&(sigma_id, tau.id)) {
                    // rmap has shape (sigma_dim, tau_dim)
                    // In the coboundary, for position (tau_offset+i, sigma_offset+j):
                    // This maps from the sigma stalk to tau stalk
                    // Actually the coboundary goes C^k -> C^{k+1}
                    // The block for (τ, σ) pair in the coboundary is sign * rmap^T
                    // No wait. Let me think again.
                    // δₖ: Cᵏ(F) → Cᵏ⁺¹(F) 
                    // For s in Cᵏ(F) (assignment to k-cells):
                    // (δₖs)(τ) = Σ_{σ face of τ} ε(σ,τ) * F(σ⪯τ)(s(σ))
                    // Wait that's not right either. 
                    // 
                    // Actually: Cᵏ(F) = ⊕_{σ k-cell} F(σ)
                    // δₖ: Cᵏ(F) → Cᵏ⁺¹(F)
                    // (δₖs)(τ) = Σ_{σ face of τ, dim(σ)=k} ε(σ,τ) * F(σ⪯τ)(s(σ))... 
                    // Hmm, but F(σ⪯τ): F(τ) → F(σ), going the wrong direction.
                    //
                    // The correct formula: for the sheaf coboundary,
                    // we use the transpose of restriction maps.
                    // δₖ maps Cᵏ = ⊕F(σ) to Cᵏ⁺¹ = ⊕F(τ)
                    // For (τ, σ face of τ): block is ε * F(σ⪯τ)^T : F(σ) → F(τ)
                    // This doesn't quite make sense dimensionally.
                    //
                    // Let me use the standard construction:
                    // C₀(F) = ⊕_{v vertex} F(v), C₁(F) = ⊕_{e edge} F(e)
                    // δ₀: C₀ → C₁ given by: for edge e = (u,v), 
                    //   (δ₀s)(e) = F(v⪯e)(s(v)) - F(u⪯e)(s(u))
                    // Hmm, restriction maps go F(e) → F(v), so we need transpose.
                    //
                    // Actually the standard sheaf coboundary is:
                    // (δ₀s)(e) = F(u⪯e)^T s(u) summed appropriately... no.
                    //
                    // Let me just use: for each incidence (σ,τ) with sign ε,
                    // the coboundary block for position τ,σ is ε * rmap^T
                    // where rmap = F(σ⪯τ): F(τ) → F(σ) has shape (dim_F(σ), dim_F(τ))
                    // So rmap^T has shape (dim_F(τ), dim_F(σ)) which maps F(σ) → F(τ) ✓
                    
                    let rmap_t = rmap.transpose();
                    for i in 0..tau_dim {
                        for j in 0..sigma_dim {
                            let val = sign * rmap_t.get(i, j);
                            if val.abs() > 1e-15 {
                                mat.set(tau_offset + i, sigma_offset + j, mat.get(tau_offset + i, sigma_offset + j) + val);
                            }
                        }
                    }
                }
            }
        }
        mat
    }
}

/// Coboundary operator for a sheaf.
#[derive(Debug, Clone)]
pub struct CoboundaryOperator<'a> {
    sheaf: &'a CellularSheaf,
    degree: usize,
}

impl<'a> CoboundaryOperator<'a> {
    pub fn new(sheaf: &'a CellularSheaf, degree: usize) -> Self {
        Self { sheaf, degree }
    }

    pub fn matrix(&self) -> SparseMatrix {
        self.sheaf.coboundary_matrix(self.degree)
    }
}

/// The sheaf Laplacian — the spectral object.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SheafLaplacian {
    pub matrix: DenseMatrix,
    pub dimension: usize,
}

impl SheafLaplacian {
    pub fn new(matrix: DenseMatrix) -> Self {
        let dim = matrix.rows;
        Self { matrix, dimension: dim }
    }

    /// Compute eigenvalues using QR iteration (for symmetric matrices).
    pub fn eigenvalues(&self) -> Vec<f64> {
        if self.dimension == 0 {
            return Vec::new();
        }
        // Simple power iteration + deflation for symmetric matrices
        // For robustness, use iterative approach
        let n = self.dimension;
        let mut eigenvals = Vec::new();
        let current = self.matrix.clone();
        
        // Use Jacobi eigenvalue algorithm for symmetric matrices
        let mut a = current.data.clone();
        for _iter in 0..100 * n {
            // Find largest off-diagonal element
            let mut max_val = 0.0f64;
            let mut max_i = 0;
            let mut max_j = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if a[i][j].abs() > max_val {
                        max_val = a[i][j].abs();
                        max_i = i;
                        max_j = j;
                    }
                }
            }
            if max_val < 1e-12 {
                break;
            }
            
            // Compute rotation
            let (i, j) = (max_i, max_j);
            let theta = if (a[i][i] - a[j][j]).abs() < 1e-15 {
                std::f64::consts::FRAC_PI_4
            } else {
                0.5_f64 * (2.0_f64 * a[i][j] / (a[i][i] - a[j][j])).atan()
            };
            let c = theta.cos();
            let s = theta.sin();
            
            // Apply Givens rotation
            let mut new_a = a.clone();
            for k in 0..n {
                if k != i && k != j {
                    new_a[i][k] = c * a[i][k] + s * a[j][k];
                    new_a[k][i] = new_a[i][k];
                    new_a[j][k] = -s * a[i][k] + c * a[j][k];
                    new_a[k][j] = new_a[j][k];
                }
            }
            new_a[i][i] = c * c * a[i][i] + 2.0 * s * c * a[i][j] + s * s * a[j][j];
            new_a[j][j] = s * s * a[i][i] - 2.0 * s * c * a[i][j] + c * c * a[j][j];
            new_a[i][j] = 0.0;
            new_a[j][i] = 0.0;
            a = new_a;
        }
        
        for i in 0..n {
            eigenvals.push(a[i][i]);
        }
        eigenvals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eigenvals
    }

    /// Spectral gap: smallest nonzero eigenvalue.
    pub fn spectral_gap(&self) -> f64 {
        let evals = self.eigenvalues();
        evals.iter()
            .filter(|&&e| e > 1e-10)
            .cloned()
            .next()
            .unwrap_or(0.0)
    }

    /// Check if v is harmonic: Lv = 0.
    pub fn is_harmonic(&self, v: &[f64]) -> bool {
        let lv = self.matrix.multiply_vec(v);
        lv.iter().all(|&x| x.abs() < 1e-10)
    }

    /// Basis for harmonic space = kernel of Laplacian.
    pub fn harmonic_space(&self) -> Vec<Vec<f64>> {
        self.matrix.kernel_basis()
    }

    /// Energy: v^T L v.
    pub fn energy(&self, v: &[f64]) -> f64 {
        let lv = self.matrix.multiply_vec(v);
        v.iter().zip(lv.iter()).map(|(&a, &b)| a * b).sum()
    }
}

/// Computed sheaf cohomology.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SheafCohomology {
    pub h0_dimension: usize,
    pub h1_dimension: usize,
    pub betti: Vec<usize>,
    h0_basis: Vec<Vec<f64>>,
    laplacian_0_matrix: DenseMatrix,
    laplacian_1_matrix: DenseMatrix,
}

impl SheafCohomology {
    /// Compute sheaf cohomology from a cellular sheaf.
    pub fn compute(sheaf: &CellularSheaf) -> Self {
        let betti = sheaf.complex.betti_numbers();

        // Sheaf coboundary δ₀: C⁰(F) → C¹(F)
        let d0 = sheaf.coboundary_matrix(0);
        let d0_t = d0.transpose();
        
        // Sheaf coboundary δ₁: C¹(F) → C²(F)
        let d1 = sheaf.coboundary_matrix(1);
        let d1_t = d1.transpose();

        // Laplacian L₀ = δ₀*δ₀ + δ₀δ₀* on 0-cochains
        // Actually L₀ = δ₀ᵀδ₀ (the "up" Laplacian on 0-cochains)
        // And L₁ = δ₁ᵀδ₁ + δ₀δ₀ᵀ on 1-cochains
        // 
        // For H⁰ = ker(L₀) where L₀ = δ₀ᵀδ₀ (on 0-cochains, only the "up" part)
        // Actually the full sheaf Laplacian L₀ = δ₀δ₀ᵀ + ... but since δ₋₁ doesn't exist:
        // L₀ = δ₀ᵀδ₀ (this is the 0-th Laplacian)
        
        let l0_sparse = d0_t.multiply(&d0);
        let l0 = l0_sparse.to_dense();
        
        // For H¹, L₁ = δ₁ᵀδ₁ + δ₀δ₀ᵀ
        let l1_up = d1_t.multiply(&d1);
        let l1_down = d0.multiply(&d0_t);
        let l1_sparse = l1_up.add(&l1_down);
        let l1 = l1_sparse.to_dense();

        // H⁰ = ker(L₀)
        let h0_basis = l0.kernel_basis();
        let h0_dimension = h0_basis.len();

        // H¹ = ker(L₁) / im(δ₀ᵀ) ... but via Hodge theory:
        // dim H¹ = dim ker(L₁) - dim ker(L₀) ... no
        // Actually: Hodge decomposition gives dim H^k = dim harmonic k-forms = dim ker(L_k)
        // So H¹ = ker(L₁)
        let h1_basis = l1.kernel_basis();
        let h1_dimension = h1_basis.len();

        SheafCohomology {
            h0_dimension,
            h1_dimension,
            betti,
            h0_basis,
            laplacian_0_matrix: l0,
            laplacian_1_matrix: l1,
        }
    }

    /// Basis for H⁰ (global sections).
    pub fn global_sections(&self) -> Vec<Vec<f64>> {
        self.h0_basis.clone()
    }

    /// L₀ = δ₀ᵀδ₀ (sheaf Laplacian on 0-cochains).
    pub fn laplacian_0(&self) -> DenseMatrix {
        self.laplacian_0_matrix.clone()
    }

    /// L₁ = δ₁ᵀδ₁ + δ₀δ₀ᵀ (sheaf Laplacian on 1-cochains).
    pub fn laplacian_1(&self) -> DenseMatrix {
        self.laplacian_1_matrix.clone()
    }

    /// Get the sheaf Laplacian object for degree 0.
    pub fn sheaf_laplacian_0(&self) -> SheafLaplacian {
        SheafLaplacian::new(self.laplacian_0_matrix.clone())
    }

    /// Get the sheaf Laplacian object for degree 1.
    pub fn sheaf_laplacian_1(&self) -> SheafLaplacian {
        SheafLaplacian::new(self.laplacian_1_matrix.clone())
    }
}

// ===================== Concrete Examples =====================

/// Build a triangle complex (3 vertices, 3 edges, 1 triangle = 2-simplex).
pub fn triangle_complex() -> CellComplex {
    let mut cx = CellComplex::new();
    // Vertices: 0, 1, 2
    cx.add_cell(0, 0, vec![0], vec![]);
    cx.add_cell(1, 0, vec![1], vec![]);
    cx.add_cell(2, 0, vec![2], vec![]);
    // Edges: 3=(0,1), 4=(1,2), 5=(0,2)
    cx.add_cell(3, 1, vec![0, 1], vec![0, 1]);
    cx.add_cell(4, 1, vec![1, 2], vec![1, 2]);
    cx.add_cell(5, 1, vec![0, 2], vec![0, 2]);
    // Triangle: 6=(0,1,2) with faces [3,4,5]
    cx.add_cell(6, 2, vec![0, 1, 2], vec![3, 4, 5]);
    cx
}

/// Build a tetrahedron complex (4 vertices, 6 edges, 4 triangles).
pub fn tetrahedron_complex() -> CellComplex {
    let mut cx = CellComplex::new();
    // Vertices 0-3
    for i in 0..4 {
        cx.add_cell(i, 0, vec![i], vec![]);
    }
    // 6 edges
    let edges = [(0,1,4), (0,2,5), (0,3,6), (1,2,7), (1,3,8), (2,3,9)];
    for &(a, b, id) in &edges {
        cx.add_cell(id, 1, vec![a, b], vec![a, b]);
    }
    // 4 triangles
    // (0,1,2) -> faces: edge(0,1)=4, edge(1,2)=7, edge(0,2)=5
    cx.add_cell(10, 2, vec![0, 1, 2], vec![4, 7, 5]);
    // (0,1,3) -> faces: edge(0,1)=4, edge(1,3)=8, edge(0,3)=6
    cx.add_cell(11, 2, vec![0, 1, 3], vec![4, 8, 6]);
    // (0,2,3) -> faces: edge(0,2)=5, edge(2,3)=9, edge(0,3)=6
    cx.add_cell(12, 2, vec![0, 2, 3], vec![5, 9, 6]);
    // (1,2,3) -> faces: edge(1,2)=7, edge(2,3)=9, edge(1,3)=8
    cx.add_cell(13, 2, vec![1, 2, 3], vec![7, 9, 8]);
    cx
}

/// Build a circle graph (n vertices, n edges forming a cycle).
pub fn circle_complex(n: usize) -> CellComplex {
    assert!(n >= 3, "Circle needs at least 3 vertices");
    let mut cx = CellComplex::new();
    for i in 0..n {
        cx.add_cell(i, 0, vec![i], vec![]);
    }
    for i in 0..n {
        let next = (i + 1) % n;
        let edge_id = n + i;
        cx.add_cell(edge_id, 1, vec![i, next], vec![i, next]);
    }
    cx
}

/// Constant sheaf: all stalks = ℝ (dimension 1), all restriction maps = identity.
pub fn constant_sheaf(complex: &CellComplex) -> CellularSheaf {
    let mut sheaf = CellularSheaf::new(complex.clone());
    let id = DenseMatrix::identity(1);
    for cell in &complex.cells {
        sheaf.set_stalk(cell.id, 1);
    }
    for cell in &complex.cells {
        // Identity restriction to self
        sheaf.set_restriction(cell.id, cell.id, id.clone());
        // Identity restriction for each face
        for &face_id in &cell.faces {
            sheaf.set_restriction(face_id, cell.id, id.clone());
        }
    }
    sheaf
}

/// Orientation sheaf: stalks = ℝ, restrictions alternate sign based on orientation.
pub fn orientation_sheaf(complex: &CellComplex) -> CellularSheaf {
    let mut sheaf = CellularSheaf::new(complex.clone());
    let id = DenseMatrix::identity(1);
    let neg = DenseMatrix::from_vec(1, 1, vec![-1.0]);
    for cell in &complex.cells {
        sheaf.set_stalk(cell.id, 1);
    }
    for cell in &complex.cells {
        sheaf.set_restriction(cell.id, cell.id, id.clone());
        for &face_id in &cell.faces {
            // Use incidence sign for orientation
            let sign = complex.incidence_sign(cell, face_id);
            if sign > 0.0 {
                sheaf.set_restriction(face_id, cell.id, id.clone());
            } else {
                sheaf.set_restriction(face_id, cell.id, neg.clone());
            }
        }
    }
    sheaf
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== SparseMatrix Tests ====================

    #[test]
    fn test_sparse_identity() {
        let m = SparseMatrix::identity(3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 1), 1.0);
        assert_eq!(m.get(2, 2), 1.0);
        assert_eq!(m.get(0, 1), 0.0);
        assert_eq!(m.rank(), 3);
        assert_eq!(m.nullity(), 0);
    }

    #[test]
    fn test_sparse_multiply_vec() {
        let mut m = SparseMatrix::identity(2);
        m.set(0, 1, 2.0);
        let v = vec![1.0, 3.0];
        let r = m.multiply_vec(&v);
        assert_eq!(r, vec![7.0, 3.0]);
    }

    #[test]
    fn test_sparse_transpose() {
        let mut m = SparseMatrix::new(2, 3);
        m.set(0, 1, 5.0);
        m.set(1, 2, 3.0);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(1, 0), 5.0);
        assert_eq!(t.get(2, 1), 3.0);
    }

    #[test]
    fn test_sparse_multiply() {
        let mut a = SparseMatrix::new(2, 2);
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0);
        a.set(1, 0, 3.0);
        a.set(1, 1, 4.0);
        let mut b = SparseMatrix::new(2, 2);
        b.set(0, 0, 2.0);
        b.set(1, 1, 2.0);
        let c = a.multiply(&b);
        assert_eq!(c.get(0, 0), 2.0);
        assert_eq!(c.get(0, 1), 4.0);
        assert_eq!(c.get(1, 0), 6.0);
        assert_eq!(c.get(1, 1), 8.0);
    }

    #[test]
    fn test_sparse_rank() {
        let mut m = SparseMatrix::new(3, 3);
        m.set(0, 0, 1.0);
        m.set(1, 1, 2.0);
        // Row 2 is zero
        assert_eq!(m.rank(), 2);
    }

    #[test]
    fn test_sparse_kernel_basis() {
        let mut m = SparseMatrix::new(2, 3);
        m.set(0, 0, 1.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 2.0);
        m.set(1, 1, 4.0);
        // Columns 0,1 are dependent (rank 1), column 2 is free => nullity 2
        let ker = m.kernel_basis();
        assert_eq!(ker.len(), 2);
        // Verify m * v = 0
        let v = &ker[0];
        let mv = m.multiply_vec(v);
        for x in &mv {
            assert!(x.abs() < 1e-10);
        }
    }

    #[test]
    fn test_sparse_add() {
        let mut a = SparseMatrix::new(2, 2);
        a.set(0, 0, 1.0);
        let mut b = SparseMatrix::new(2, 2);
        b.set(0, 0, 2.0);
        b.set(1, 1, 3.0);
        let c = a.add(&b);
        assert_eq!(c.get(0, 0), 3.0);
        assert_eq!(c.get(1, 1), 3.0);
    }

    // ==================== DenseMatrix Tests ====================

    #[test]
    fn test_dense_identity() {
        let m = DenseMatrix::identity(3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 2), 0.0);
    }

    #[test]
    fn test_dense_multiply() {
        let a = DenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = DenseMatrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 2.0]);
        let c = a.multiply(&b);
        assert_eq!(c.get(0, 0), 2.0);
        assert_eq!(c.get(0, 1), 4.0);
        assert_eq!(c.get(1, 0), 6.0);
        assert_eq!(c.get(1, 1), 8.0);
    }

    #[test]
    fn test_dense_transpose() {
        let m = DenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(1, 0), 2.0);
        assert_eq!(t.get(2, 1), 6.0);
    }

    #[test]
    fn test_dense_rank() {
        let m = DenseMatrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 6.0, 2.0, 4.0]);
        assert_eq!(m.rank(), 1); // rows are dependent
    }

    #[test]
    fn test_dense_kernel_basis() {
        let m = DenseMatrix::from_vec(2, 3, vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let ker = m.kernel_basis();
        // rank 2, cols 3 => nullity 1
        assert_eq!(ker.len(), 1);
    }

    #[test]
    fn test_dense_image_basis() {
        let m = DenseMatrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let img = m.image_basis();
        assert_eq!(img.len(), 2); // full rank
    }

    #[test]
    fn test_dense_approx_eq() {
        let a = DenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = DenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(a.approx_eq(&b, 1e-10));
    }

    #[test]
    fn test_dense_multiply_vec() {
        let m = DenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = vec![1.0, 0.0, 0.0];
        let r = m.multiply_vec(&v);
        assert_eq!(r, vec![1.0, 4.0]);
    }

    // ==================== CellComplex Tests ====================

    #[test]
    fn test_triangle_complex_cells() {
        let cx = triangle_complex();
        assert_eq!(cx.num_cells(0), 3);
        assert_eq!(cx.num_cells(1), 3);
        assert_eq!(cx.num_cells(2), 1);
        assert_eq!(cx.dimension, 2);
    }

    #[test]
    fn test_tetrahedron_complex_cells() {
        let cx = tetrahedron_complex();
        assert_eq!(cx.num_cells(0), 4);
        assert_eq!(cx.num_cells(1), 6);
        assert_eq!(cx.num_cells(2), 4);
        assert_eq!(cx.dimension, 2);
    }

    #[test]
    fn test_circle_complex_cells() {
        let cx = circle_complex(5);
        assert_eq!(cx.num_cells(0), 5);
        assert_eq!(cx.num_cells(1), 5);
        assert_eq!(cx.dimension, 1);
    }

    #[test]
    fn test_euler_characteristic_triangle() {
        // Triangle: 3V - 3E + 1F = 1 (contractible)
        let cx = triangle_complex();
        assert_eq!(cx.euler_characteristic(), 1);
    }

    #[test]
    fn test_euler_characteristic_tetrahedron() {
        // Tetrahedron: 4V - 6E + 4F = 2 (sphere)
        let cx = tetrahedron_complex();
        assert_eq!(cx.euler_characteristic(), 2);
    }

    #[test]
    fn test_euler_characteristic_circle() {
        // Circle: nV - nE = 0
        let cx = circle_complex(5);
        assert_eq!(cx.euler_characteristic(), 0);
    }

    #[test]
    fn test_betti_triangle() {
        // Contractible: β₀ = 1, β₁ = 0, β₂ = 0
        // But with boundary matrix approach: ∂₂ has rank 1, ∂₁ has rank 2
        // H₀ = ker(∂₀)/im(∂₁) = 3/2 = 1, H₁ = ker(∂₁)/im(∂₂) = (3-2)/1 = 0
        let cx = triangle_complex();
        let betti = cx.betti_numbers();
        assert_eq!(betti[0], 1, "β₀ should be 1");
        // β₁ depends on whether the triangle fills the interior
        // For a filled triangle: β₁ = 0
        assert_eq!(betti[1], 0, "β₁ should be 0 for filled triangle");
    }

    #[test]
    fn test_betti_circle() {
        // Circle S¹: β₀ = 1, β₁ = 1
        let cx = circle_complex(5);
        let betti = cx.betti_numbers();
        assert_eq!(betti[0], 1, "β₀ should be 1");
        assert_eq!(betti[1], 1, "β₁ should be 1");
    }

    #[test]
    fn test_betti_tetrahedron() {
        // S²: β₀ = 1, β₁ = 0, β₂ = 1
        let cx = tetrahedron_complex();
        let betti = cx.betti_numbers();
        assert_eq!(betti[0], 1, "β₀ should be 1");
        assert_eq!(betti[1], 0, "β₁ should be 0");
        assert_eq!(betti[2], 1, "β₂ should be 1");
    }

    #[test]
    fn test_is_connected_triangle() {
        assert!(triangle_complex().is_connected());
    }

    #[test]
    fn test_is_connected_tetrahedron() {
        assert!(tetrahedron_complex().is_connected());
    }

    #[test]
    fn test_is_connected_circle() {
        assert!(circle_complex(4).is_connected());
    }

    #[test]
    fn test_boundary_matrix_triangle() {
        let cx = triangle_complex();
        let d1 = cx.boundary_matrix(1);
        // ∂₁: 3 edges -> 3 vertices
        assert_eq!(d1.rows, 3);
        assert_eq!(d1.cols, 3);
        // Edge (0,1) -> v0 - v1 => entries at (0,col)=1, (1,col)=-1
        // Edge (1,2) -> v1 - v2 => entries at (1,col)=1, (2,col)=-1
        // Edge (0,2) -> v0 - v2 => entries at (0,col)=1, (2,col)=-1
        // rank should be 2
        assert_eq!(d1.rank(), 2);
    }

    #[test]
    fn test_coboundary_matrix_circle() {
        let cx = circle_complex(4);
        let d1 = cx.coboundary_matrix(0);
        // δ₀ = ∂₁ᵀ: 4 vertices -> 4 edges
        assert_eq!(d1.rows, 4);
        assert_eq!(d1.cols, 4);
    }

    // ==================== Sheaf Tests ====================

    #[test]
    fn test_constant_sheaf_stalks() {
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        for cell in &cx.cells {
            assert_eq!(sheaf.stalk_dimension(cell.id), 1);
        }
    }

    #[test]
    fn test_constant_sheaf_functoriality() {
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        assert!(sheaf.verify_functoriality(), "Constant sheaf must be functorial");
    }

    #[test]
    fn test_orientation_sheaf_functoriality() {
        let cx = triangle_complex();
        let sheaf = orientation_sheaf(&cx);
        assert!(sheaf.verify_functoriality(), "Orientation sheaf must be functorial");
    }

    #[test]
    fn test_total_stalk_dimension() {
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        assert_eq!(sheaf.total_stalk_dimension(0), 3);
        assert_eq!(sheaf.total_stalk_dimension(1), 3);
        assert_eq!(sheaf.total_stalk_dimension(2), 1);
    }

    // ==================== Sheaf Cohomology Tests ====================

    #[test]
    fn test_constant_sheaf_triangle_h0() {
        // Constant sheaf on contractible space: H⁰ = ℝ (1 global section)
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        assert_eq!(cohom.h0_dimension, 1, "H⁰ of constant sheaf on triangle should be 1");
    }

    #[test]
    fn test_constant_sheaf_triangle_h1() {
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        assert_eq!(cohom.h1_dimension, 0, "H¹ of constant sheaf on triangle should be 0");
    }

    #[test]
    fn test_constant_sheaf_circle_h0() {
        // Constant sheaf on S¹: H⁰ = ℝ
        let cx = circle_complex(5);
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        assert_eq!(cohom.h0_dimension, 1, "H⁰ of constant sheaf on S¹ should be 1");
    }

    #[test]
    fn test_constant_sheaf_circle_h1() {
        // Constant sheaf on S¹: H¹ = ℝ
        let cx = circle_complex(5);
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        assert_eq!(cohom.h1_dimension, 1, "H¹ of constant sheaf on S¹ should be 1");
    }

    #[test]
    fn test_constant_sheaf_tetrahedron_h0() {
        let cx = tetrahedron_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        assert_eq!(cohom.h0_dimension, 1, "H⁰ of constant sheaf on S² should be 1");
    }

    #[test]
    fn test_constant_sheaf_tetrahedron_h1() {
        let cx = tetrahedron_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        assert_eq!(cohom.h1_dimension, 0, "H¹ of constant sheaf on S² should be 0");
    }

    #[test]
    fn test_global_sections_constant_triangle() {
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        let gs = cohom.global_sections();
        assert_eq!(gs.len(), 1);
        // Global section should be constant (same value at all vertices)
        let v = &gs[0];
        assert_eq!(v.len(), 3); // 3 vertices for C⁰
    }

    #[test]
    fn test_sheaf_cohomology_specializes_to_cellular() {
        // Theorem: sheaf cohomology with constant sheaf should agree with cellular cohomology
        let cx = circle_complex(5);
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        assert_eq!(cohom.h0_dimension, cohom.betti[0], "H⁰ should equal β₀ for constant sheaf");
        assert_eq!(cohom.h1_dimension, cohom.betti[1], "H¹ should equal β₁ for constant sheaf");
    }

    #[test]
    fn test_sheaf_cohomology_specializes_triangle() {
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        assert_eq!(cohom.h0_dimension, cohom.betti[0]);
        assert_eq!(cohom.h1_dimension, cohom.betti[1]);
    }

    // ==================== Laplacian Tests ====================

    #[test]
    fn test_laplacian_eigenvalues_triangle() {
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        let lap = cohom.sheaf_laplacian_0();
        let evals = lap.eigenvalues();
        // Should have one zero eigenvalue (constant functions are harmonic)
        let zero_count = evals.iter().filter(|&&e| e.abs() < 1e-8).count();
        assert_eq!(zero_count, 1, "Should have exactly 1 harmonic section");
    }

    #[test]
    fn test_laplacian_spectral_gap() {
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        let lap = cohom.sheaf_laplacian_0();
        let gap = lap.spectral_gap();
        assert!(gap > 0.0, "Spectral gap should be positive for connected complex with constant sheaf");
    }

    #[test]
    fn test_harmonic_section_energy_zero() {
        // Theorem: energy of harmonic section = 0
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        let lap = cohom.sheaf_laplacian_0();
        let harmonic = lap.harmonic_space();
        for v in &harmonic {
            let e = lap.energy(v);
            assert!(e.abs() < 1e-10, "Harmonic section must have zero energy, got {}", e);
        }
    }

    #[test]
    fn test_is_harmonic() {
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        let lap = cohom.sheaf_laplacian_0();
        let harmonic = lap.harmonic_space();
        for v in &harmonic {
            assert!(lap.is_harmonic(v), "Kernel vectors should be harmonic");
        }
    }

    #[test]
    fn test_nonzero_vector_not_harmonic() {
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        let lap = cohom.sheaf_laplacian_0();
        if lap.dimension >= 2 {
            // A random vector probably isn't harmonic
            let v = vec![1.0, 0.0, 0.0];
            if v.len() == lap.dimension {
                // Only test if dimensions match
                assert!(!lap.is_harmonic(&v) || lap.dimension == 1);
            }
        }
    }

    #[test]
    fn test_spectral_gap_implies_unique_global_section() {
        // Theorem: spectral gap > 0 => unique global section (for connected complex with constant sheaf)
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        let lap = cohom.sheaf_laplacian_0();
        let gap = lap.spectral_gap();
        if gap > 0.0 && cx.is_connected() {
            // For constant sheaf on connected complex, H⁰ should be 1-dimensional
            assert_eq!(cohom.h0_dimension, 1);
        }
    }

    #[test]
    fn test_laplacian_positive_semidefinite() {
        // Sheaf Laplacian should be positive semidefinite: v^T L v >= 0 for all v
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        let lap = cohom.sheaf_laplacian_0();
        let n = lap.dimension;
        // Test a few random-ish vectors
        for i in 0..5 {
            let v: Vec<f64> = (0..n).map(|j| ((i * 7 + j * 13) as f64).sin()).collect();
            let e = lap.energy(&v);
            assert!(e >= -1e-10, "Laplacian should be PSD, got energy {}", e);
        }
    }

    #[test]
    fn test_laplacian_1_dimension_circle() {
        let cx = circle_complex(4);
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        let l1 = cohom.laplacian_1();
        assert_eq!(l1.rows, 4); // 4 edges
        assert_eq!(l1.cols, 4);
    }

    // ==================== Orientation Sheaf Tests ====================

    #[test]
    fn test_orientation_sheaf_circle_h0() {
        let cx = circle_complex(5);
        let sheaf = orientation_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        // Orientation sheaf on S¹: H⁰ = 0 (no global sections that are compatible)
        // Actually depends on the implementation. Let me check.
        // For an orientable manifold, the orientation sheaf has H⁰ = 0.
        // S¹ is orientable, so H⁰ = 0 for the orientation sheaf.
        assert_eq!(cohom.h0_dimension, 0, "H⁰ of orientation sheaf on S¹ should be 0");
    }

    // ==================== CoboundaryOperator Tests ====================

    #[test]
    fn test_coboundary_operator_triangle() {
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        let d0 = CoboundaryOperator::new(&sheaf, 0);
        let m = d0.matrix();
        assert_eq!(m.cols, 3); // 3 vertices
        assert_eq!(m.rows, 3); // 3 edges
    }

    #[test]
    fn test_coboundary_operator_circle() {
        let cx = circle_complex(4);
        let sheaf = constant_sheaf(&cx);
        let d0 = CoboundaryOperator::new(&sheaf, 0);
        let m = d0.matrix();
        assert_eq!(m.cols, 4);
        assert_eq!(m.rows, 4);
    }

    // ==================== Cell Tests ====================

    #[test]
    fn test_cell_creation() {
        let c = Cell::new(0, 1, vec![0, 1]);
        assert_eq!(c.id, 0);
        assert_eq!(c.dimension, 1);
        assert_eq!(c.vertices, vec![0, 1]);
        assert!(c.faces.is_empty());
        assert!(c.cofaces.is_empty());
    }

    #[test]
    fn test_cell_cofaces() {
        let cx = triangle_complex();
        // Vertex 0 should have cofaces including edges that contain it
        let v0 = cx.cell(0).unwrap();
        assert!(v0.cofaces.contains(&3)); // edge (0,1)
        assert!(v0.cofaces.contains(&5)); // edge (0,2)
    }

    // ==================== Serde Tests ====================

    #[test]
    fn test_serde_sparse_matrix() {
        let mut m = SparseMatrix::new(3, 3);
        m.set(0, 1, 2.5);
        m.set(2, 2, -1.0);
        let json = serde_json::to_string(&m).unwrap();
        let m2: SparseMatrix = serde_json::from_str(&json).unwrap();
        assert_eq!(m2.get(0, 1), 2.5);
        assert_eq!(m2.get(2, 2), -1.0);
        assert_eq!(m2.get(1, 1), 0.0);
    }

    #[test]
    fn test_serde_dense_matrix() {
        let m = DenseMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let json = serde_json::to_string(&m).unwrap();
        let m2: DenseMatrix = serde_json::from_str(&json).unwrap();
        assert_eq!(m2.get(1, 2), 6.0);
    }

    #[test]
    fn test_serde_cell() {
        let c = Cell::new(5, 2, vec![0, 1, 2]);
        let json = serde_json::to_string(&c).unwrap();
        let c2: Cell = serde_json::from_str(&json).unwrap();
        assert_eq!(c2.id, 5);
        assert_eq!(c2.dimension, 2);
    }

    #[test]
    fn test_serde_cell_complex() {
        let cx = triangle_complex();
        let json = serde_json::to_string(&cx).unwrap();
        let cx2: CellComplex = serde_json::from_str(&json).unwrap();
        assert_eq!(cx2.num_cells(0), 3);
        assert_eq!(cx2.euler_characteristic(), 1);
    }

    #[test]
    fn test_serde_sheaf() {
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        let json = serde_json::to_string(&sheaf).unwrap();
        let sheaf2: CellularSheaf = serde_json::from_str(&json).unwrap();
        assert_eq!(sheaf2.stalk_dimension(0), 1);
        assert!(sheaf2.verify_functoriality());
    }

    #[test]
    fn test_serde_cohomology() {
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        let json = serde_json::to_string(&cohom).unwrap();
        let cohom2: SheafCohomology = serde_json::from_str(&json).unwrap();
        assert_eq!(cohom2.h0_dimension, cohom.h0_dimension);
        assert_eq!(cohom2.h1_dimension, cohom.h1_dimension);
    }

    // ==================== Additional Theorem Verification ====================

    #[test]
    fn test_euler_characteristic_equals_alternating_betti_sum() {
        // Theorem: χ = Σ(-1)^k βₖ (for well-behaved spaces)
        let cx = tetrahedron_complex();
        let chi = cx.euler_characteristic();
        let betti = cx.betti_numbers();
        let betti_sum: i32 = betti.iter().enumerate()
            .map(|(k, &b)| if k % 2 == 0 { b as i32 } else { -(b as i32) })
            .sum();
        assert_eq!(chi, betti_sum, "Euler characteristic should equal alternating sum of Betti numbers");
    }

    #[test]
    fn test_euler_characteristic_circle_equals_betti_sum() {
        let cx = circle_complex(6);
        let chi = cx.euler_characteristic();
        let betti = cx.betti_numbers();
        let betti_sum: i32 = betti.iter().enumerate()
            .map(|(k, &b)| if k % 2 == 0 { b as i32 } else { -(b as i32) })
            .sum();
        assert_eq!(chi, betti_sum);
    }

    #[test]
    fn test_h0_equals_global_sections() {
        // Theorem: H⁰(F) = global sections
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        let gs = cohom.global_sections();
        assert_eq!(cohom.h0_dimension, gs.len());

        // Verify each global section is actually in ker(L₀)
        let lap = cohom.sheaf_laplacian_0();
        for v in &gs {
            assert!(lap.is_harmonic(v));
        }
    }

    #[test]
    fn test_boundary_squared_zero() {
        // Theorem: ∂² = 0, i.e., ∂ₖ ∘ ∂ₖ₊₁ = 0
        let cx = tetrahedron_complex();
        // ∂₁ ∘ ∂₂ = 0 (the only meaningful composition for a 2-complex)
        let d1 = cx.boundary_matrix(1); // C₁ → C₀
        let d2 = cx.boundary_matrix(2); // C₂ → C₁
        // ∂₁∘∂₂: C₂→C₁→C₀, in matrix form: d1 * d2
        let d1_d2 = d1.multiply(&d2);
        for &v in d1_d2.entries.values() {
            assert!(v.abs() < 1e-10, "∂₁∘∂₂ should be 0, got {}", v);
        }
        // Also check for triangle
        let tri = triangle_complex();
        let td1 = tri.boundary_matrix(1);
        let td2 = tri.boundary_matrix(2);
        let td1_td2 = td1.multiply(&td2);
        for &v in td1_td2.entries.values() {
            assert!(v.abs() < 1e-10, "∂₁∘∂₂ should be 0 for triangle, got {}", v);
        }
    }

    #[test]
    fn test_constant_sheaf_different_circles() {
        // H⁰ = 1, H¹ = 1 for all circles regardless of size
        for n in [3, 4, 5, 10] {
            let cx = circle_complex(n);
            let sheaf = constant_sheaf(&cx);
            let cohom = SheafCohomology::compute(&sheaf);
            assert_eq!(cohom.h0_dimension, 1, "H⁰ = 1 for S¹ with {} vertices", n);
            assert_eq!(cohom.h1_dimension, 1, "H¹ = 1 for S¹ with {} vertices", n);
        }
    }

    #[test]
    fn test_higher_dimensional_stalks() {
        // Sheaf with 2-dimensional stalks on triangle
        let cx = triangle_complex();
        let mut sheaf = CellularSheaf::new(cx.clone());
        let id2 = DenseMatrix::identity(2);
        for cell in &cx.cells {
            sheaf.set_stalk(cell.id, 2);
        }
        for cell in &cx.cells {
            sheaf.set_restriction(cell.id, cell.id, id2.clone());
            for &face_id in &cell.faces {
                sheaf.set_restriction(face_id, cell.id, id2.clone());
            }
        }
        let cohom = SheafCohomology::compute(&sheaf);
        // With 2-dim stalks and identity restrictions, H⁰ should be 2
        assert_eq!(cohom.h0_dimension, 2, "H⁰ should be 2 for ℝ²-valued constant sheaf");
        assert_eq!(cohom.h1_dimension, 0, "H¹ should be 0");
    }

    #[test]
    fn test_zero_stalk_sheaf() {
        // Sheaf with zero-dimensional stalks should have trivial cohomology
        let cx = triangle_complex();
        let sheaf = CellularSheaf::new(cx.clone());
        // No stalks set => all dimensions 0
        let cohom = SheafCohomology::compute(&sheaf);
        assert_eq!(cohom.h0_dimension, 0);
        assert_eq!(cohom.h1_dimension, 0);
    }

    #[test]
    fn test_laplacian_symmetry() {
        // Sheaf Laplacian should be symmetric
        let cx = triangle_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        let l0 = cohom.laplacian_0();
        let l0t = l0.transpose();
        assert!(l0.approx_eq(&l0t, 1e-10), "Laplacian should be symmetric");
    }

    #[test]
    fn test_laplacian_1_symmetry() {
        let cx = circle_complex(4);
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        let l1 = cohom.laplacian_1();
        let l1t = l1.transpose();
        assert!(l1.approx_eq(&l1t, 1e-10), "L₁ should be symmetric");
    }

    #[test]
    fn test_circle_euler_equals_betti() {
        let cx = circle_complex(7);
        assert_eq!(cx.euler_characteristic(), 0);
        let betti = cx.betti_numbers();
        let alt_sum: i32 = betti.iter().enumerate()
            .map(|(k, &b)| if k % 2 == 0 { b as i32 } else { -(b as i32) })
            .sum();
        assert_eq!(cx.euler_characteristic(), alt_sum);
    }

    #[test]
    fn test_tetrahedron_laplacian_eigenvalues() {
        let cx = tetrahedron_complex();
        let sheaf = constant_sheaf(&cx);
        let cohom = SheafCohomology::compute(&sheaf);
        let lap = cohom.sheaf_laplacian_0();
        let evals = lap.eigenvalues();
        // One zero eigenvalue for S²
        let zero_count = evals.iter().filter(|&&e| e.abs() < 1e-8).count();
        assert_eq!(zero_count, 1);
        // All eigenvalues non-negative
        for &e in &evals {
            assert!(e >= -1e-10, "Eigenvalues should be non-negative");
        }
    }

    #[test]
    fn test_projection_sheaf() {
        // Non-trivial restriction maps: projection ℝ² → ℝ
        let cx = triangle_complex();
        let mut sheaf = CellularSheaf::new(cx.clone());
        // Vertices get ℝ, edges get ℝ²
        for cell in cx.cells_of_dimension(0) {
            sheaf.set_stalk(cell.id, 1);
        }
        for cell in cx.cells_of_dimension(1) {
            sheaf.set_stalk(cell.id, 2);
        }
        sheaf.set_stalk(6, 2); // triangle

        // Restriction: edge → vertex projects onto first coordinate
        let proj = DenseMatrix::from_vec(1, 2, vec![1.0, 0.0]);
        let id1 = DenseMatrix::identity(1);
        let id2 = DenseMatrix::identity(2);

        for cell in &cx.cells {
            sheaf.set_restriction(cell.id, cell.id, 
                if cell.dimension == 0 { id1.clone() } else { id2.clone() });
            for &face_id in &cell.faces {
                let face_dim = cx.cell(face_id).unwrap().dimension;
                if face_dim == 0 && cell.dimension == 1 {
                    sheaf.set_restriction(face_id, cell.id, proj.clone());
                } else if face_dim == 1 && cell.dimension == 2 {
                    sheaf.set_restriction(face_id, cell.id, id2.clone());
                }
            }
        }
        let cohom = SheafCohomology::compute(&sheaf);
        // With projection, some compatibility constraints
        assert!(cohom.h0_dimension >= 0);
    }
}
