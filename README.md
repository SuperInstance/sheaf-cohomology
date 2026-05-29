# sheaf-cohomology

**Cellular sheaves and cohomology on graphs — pure Rust, zero dependencies.**

## The Big Idea: Cohomology Measures the Gap Between Local and Global

You have a network of sensors. Each sensor measures temperature locally. Neighboring sensors can compare readings. Locally, everything is consistent — each sensor's reading agrees with its neighbors. But globally, the readings might not stitch together into a single consistent temperature field.

**Sheaf cohomology measures exactly this gap.**

- **H⁰** = the space of globally consistent assignments. Everything agrees everywhere.
- **H¹** = the obstruction. Where local consistency *cannot* be extended to global consistency. The holes in your data.

```
Three sensors in a line:           Three sensors in a triangle:

  A ──── B ──── C                   A ──── B
                                    |      |
  Local agreement propagates.       |      |
  H⁰ = 1 (one global temperature)   C ──── 
                                    Local pairs all agree,
                                    but A→B→C→A might not close.
                                    H¹ = obstruction.
```

If H⁰ is large, your system is coherent — lots of things agree globally. If H¹ is large, there's structural disagreement that no amount of local fixing can resolve.

## What's a Sheaf?

A **cellular sheaf** on a graph assigns:
1. A **vector space** (the *stalk*) to each vertex — think "data at this node"
2. A **linear map** (the *restriction*) to each edge — think "how data at one node relates to its neighbor"

```rust
use sheaf_cohomology::{Sheaf, VertexStalk, EdgeRestriction, dim_h0, dim_h1};

// A sheaf on a triangle where each vertex has 2D data
// and edges enforce identity (neighbor data must be equal)
let sheaf = Sheaf::constant(3, 2);

let h0 = dim_h0(&sheaf);  // = 2 (globally consistent: same 2D vector everywhere)
let h1 = dim_h1(&sheaf);  // = 1 (the cycle creates an obstruction)
```

For a **constant sheaf** (identity restrictions), H⁰ equals the stalk dimension on a connected graph — there's one "global opinion" per dimension. Disconnected vertices get independent opinions, so H⁰ goes up.

## The Coboundary Operator: Where Local Meets Global

The magic happens through the **coboundary** δ: C⁰ → C¹.

An element of C⁰ is an assignment of data to every vertex. The coboundary measures the *disagreement* across every edge:

```
(δs)_{edge {i,j}} = F_{ij}(s_i) − s_j
```

where F_{ij} is the restriction map from vertex i to vertex j.

```rust
use sheaf_cohomology::{Sheaf, coboundary, coboundary_adjoint};

let sheaf = Sheaf::constant(3, 2);
let delta = coboundary(&sheaf);

// δ is a matrix mapping C⁰ → C¹
// For constant(3, 2): C⁰ has dim 6 (3 vertices × 2D), C¹ has dim 6 (3 edges × 2D)
println!("C⁰ dimension: {}", delta.cols);  // 6
println!("C¹ dimension: {}", delta.rows);  // 6

// The adjoint δ* maps C¹ → C⁰ (transpose of δ)
let delta_star = coboundary_adjoint(&sheaf);
```

**The cohomology groups:**
- **H⁰ = ker(δ)** — assignments where all edges agree. Global sections.
- **H¹ = coker(δ)** — edge assignments that *cannot* come from any vertex assignment. The gap.

## The Sheaf Laplacian: L = δ* δ

Just as the graph Laplacian measures how "smooth" a function on vertices is, the **sheaf Laplacian** measures how well a sheaf assignment agrees globally.

```rust
use sheaf_cohomology::{Sheaf, sheaf_laplacian, dim_h0, global_sections_basis};

let sheaf = Sheaf::constant_path(4, 2);  // path graph, 2D stalks
let l = sheaf_laplacian(&sheaf);

// H⁰ = dim(ker L) — global sections live in the kernel
let h0 = dim_h0(&sheaf);  // = 2

// Get the actual global section basis vectors
let basis = global_sections_basis(&sheaf);
// For constant sheaf on a connected graph: basis spans "same value everywhere"
```

**Key property:** For 1-dimensional stalks with identity restrictions, the sheaf Laplacian *reduces exactly to the ordinary graph Laplacian*. The sheaf Laplacian is a strict generalization:

```
Sheaf Laplacian (with 1D stalks, identity restrictions) = Graph Laplacian
```

This means everything you know about spectral graph theory — eigenvalues, connectivity, diffusion — has a sheaf-theoretic generalization.

## Consistency Measures

### Local Consistency

How well does vertex *v* agree with its neighbors?

```rust
use sheaf_cohomology::{Sheaf, local_consistency};

let sheaf = Sheaf::constant(4, 2);
let lc = local_consistency(&sheaf, 0);  // ∈ (0, 1], 1.0 = perfect
```

Returns 1.0 for isolated vertices (no neighbors to disagree with) and decreases as the Laplacian energy at that vertex grows.

### Global Consistency

What fraction of all possible assignments are globally consistent?

```rust
use sheaf_cohomology::{Sheaf, global_consistency};

// Path graph (tree): no obstructions, but constraints reduce freedom
let sheaf_path = Sheaf::constant_path(4, 2);  // H⁰=2, C⁰=8
let gc = global_consistency(&sheaf_path);  // = 2/8 = 0.25

// Disconnected: every assignment is valid
let sheaf_disconnected = Sheaf {
    num_vertices: 3,
    stalks: vec![VertexStalk(1), VertexStalk(1), VertexStalk(1)],
    edges: vec![],
};
let gc = global_consistency(&sheaf_disconnected);  // = 1.0
```

### The Twisted Sheaf: When Restrictions Reduce Agreement

Not all restriction maps are the identity. A "twisted" sheaf uses different maps on different edges, potentially reducing the space of global sections:

```rust
use sheaf_cohomology::{Sheaf, VertexStalk, EdgeRestriction, dim_h0, global_consistency};

// Triangle with 2D stalks: two identity edges, one "flip" edge
let sheaf_twisted = Sheaf {
    num_vertices: 3,
    stalks: vec![VertexStalk(2), VertexStalk(2), VertexStalk(2)],
    edges: vec![
        EdgeRestriction { i: 0, j: 1, matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]] },
        EdgeRestriction { i: 1, j: 2, matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]] },
        EdgeRestriction { i: 0, j: 2, matrix: vec![vec![1.0, 0.0], vec![0.0, -1.0]] },
        //                                          flip second component ^^^
    ],
};

// Constant sheaf: H⁰ = 2, Twisted sheaf: H⁰ = 1
// The flip kills one dimension of global agreement
```

## The Euler Characteristic

χ = dim H⁰ − dim H¹ — a single number encoding the topological character of the sheaf.

```rust
use sheaf_cohomology::{Sheaf, euler_characteristic};

// Tree (path graph): H¹ = 0, so χ = H⁰
let sheaf = Sheaf::constant_path(4, 2);
let chi = euler_characteristic(&sheaf);  // = 2

// Cycle: H¹ ≠ 0, so χ captures the obstruction
```

For a constant sheaf on a tree: χ = stalk dimension (no cycles, no obstructions).
For a constant sheaf on a cycle: χ = stalk dimension − H¹ (the cycle creates an obstruction).

## Connection to Conservation Ratio

The sheaf Laplacian's spectral properties connect directly to the **conservation ratio** framework:

- The **smallest nonzero eigenvalue** of the sheaf Laplacian measures how strongly the sheaf forces agreement — analogous to algebraic connectivity (λ₂) in graph theory
- The ratio λ₂/λ_max for the sheaf Laplacian gives a sheaf-theoretic "coherence ratio"
- When the sheaf degenerates (restrictions become rank-deficient), this ratio drops — an early warning of structural breakdown

This makes sheaf cohomology a natural language for understanding *when* and *how* local consistency fails to produce global consistency.

## API Summary

| Type | Description |
|------|-------------|
| `Sheaf` | A cellular sheaf on a graph (stalks + restriction maps) |
| `VertexStalk(usize)` | Dimension of the vector space at a vertex |
| `EdgeRestriction` | Linear map on an edge with source/target indices and matrix |
| `Matrix` | Dense row-major matrix with rank/nullity/kernel operations |
| `SparseMatrix` | COO-like sparse matrix representation |

| Function | Description |
|----------|-------------|
| `coboundary` | δ: C⁰ → C¹ (measures edge disagreement) |
| `coboundary_adjoint` | δ*: C¹ → C⁰ (transpose of δ) |
| `sheaf_laplacian` | L = δ*δ (generalized Laplacian) |
| `dim_h0` | Dimension of global sections |
| `dim_h1` | Dimension of obstruction space |
| `euler_characteristic` | χ = H⁰ − H¹ |
| `global_sections_basis` | Basis vectors for ker(L) |
| `local_consistency` | Per-vertex agreement measure ∈ (0, 1] |
| `global_consistency` | Overall coherence ratio ∈ (0, 1] |
| `graph_laplacian` | Standard graph Laplacian (for comparison) |

## Constructors

| Method | Description |
|--------|-------------|
| `Sheaf::constant(n, k)` | Complete graph on n vertices, k-dim stalks, identity restrictions |
| `Sheaf::constant_path(n, k)` | Path graph (tree) version |

## Honest Limitations

- **Dense linear algebra.** The sheaf Laplacian is computed as a dense matrix. This limits scalability to ~100-200 vertices with small stalks. For large graphs, you'd want sparse representations.
- **Two-term complexes only.** The library computes H⁰ and H¹ but doesn't support higher cohomology groups (H², H³, ...) that arise on higher-dimensional cell complexes (simplicial complexes, CW complexes).
- **No Hodge theory.** The Hodge decomposition (harmonic, exact, coexact) is implicit but not exposed as a first-class API.
- **f64 precision only.** No exact arithmetic for rational/algebraic restriction maps.
- **No sheaf morphisms.** You can build sheaves and compute their cohomology, but you can't yet define maps between sheaves and study induced maps on cohomology.

## When to Use This

- **Network consistency analysis** — sensor networks, distributed databases, consensus protocols
- **Topological data analysis** — understanding the structure of data through local-to-global obstructions
- **Teaching** — the code is meant to be read. Every test is a small worked example.
- **Research prototyping** — quickly test whether a sheaf-theoretic model captures your phenomenon

## Installation

```toml
[dependencies]
sheaf-cohomology = "0.1"
```

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
