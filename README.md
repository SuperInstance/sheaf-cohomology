# sheaf-cohomology

Sheaf cohomology in Rust. Local data, global constraints.

Computes sheaf cohomology groups, sheaf Laplacians, spectral gaps, Betti numbers, and Euler characteristics for cellular sheaves on topological spaces.

## Quick Start

```rust
use sheaf_cohomology::*;

// Circle S¹
let circ = circle_complex(5);
let sheaf = constant_sheaf(&circ);
let cohom = SheafCohomology::compute(&sheaf);
println!("H⁰ = {}, H¹ = {}", cohom.h0_dimension, cohom.h1_dimension); // 1, 1
```

## What's Here

- **Cell complexes** — vertices, edges, faces with boundary operators
- **Cellular sheaves** — stalks and restriction maps with functoriality checks
- **Sheaf cohomology** — H⁰, H¹ via Hodge theory (kernel of sheaf Laplacian)
- **Spectral analysis** — eigenvalues, spectral gap, harmonic sections, Dirichlet energy
- **Serialization** — JSON via serde for all types

## Pre-built Spaces

```rust
let tri = triangle_complex();        // 2-simplex
let tet = tetrahedron_complex();     // S²
let circ = circle_complex(n);        // S¹ with n vertices
```

## Run

```bash
cargo run    # Demo
cargo test   # 73+ property-based tests
```

## License

MIT OR Apache-2.0
