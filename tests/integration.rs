//! Integration tests for sheaf-cohomology
//!
//! Validates core properties of cellular sheaf cohomology on graphs.

use sheaf_cohomology::*;

// 1. Constant sheaf (1-dim stalks, identity restrictions): H⁰ dimension = number of vertices
//    For a connected graph with 1-dim stalks and identity restrictions, H⁰ = 1 (not n).
//    For a completely disconnected graph (no edges), H⁰ = n.
//    We test both: connected constant sheaf has H⁰ = 1, and per-stalk H⁰ sums correctly.

#[test]
fn constant_sheaf_connected_h0_equals_1_per_stalk_dim() {
    // Complete graph on n vertices, stalk dim k → H⁰ = k (one copy of R^k globally)
    for n in [2, 3, 5] {
        for k in [1, 2, 3] {
            let sheaf = Sheaf::constant(n, k);
            let h0 = dim_h0(&sheaf);
            assert_eq!(
                h0, k,
                "Constant sheaf on {}-vertex complete graph with stalk dim {}: H⁰ should be {}",
                n, k, k
            );
        }
    }
}

#[test]
fn constant_sheaf_no_edges_h0_equals_total_stalk_dim() {
    // Disconnected: no edges, so every stalk is free → H⁰ = sum of stalk dims
    let n = 4;
    let k = 1;
    let sheaf = Sheaf {
        num_vertices: n,
        stalks: vec![VertexStalk(k); n],
        edges: vec![],
    };
    let h0 = dim_h0(&sheaf);
    assert_eq!(
        h0, n * k,
        "Disconnected constant sheaf on {} vertices: H⁰ = {}",
        n, n * k
    );
}

// 2. Trivial sheaf on 2 vertices: global sections exist

#[test]
fn trivial_sheaf_on_two_vertices_has_global_sections() {
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
    assert!(
        h0 > 0,
        "Trivial sheaf on 2 vertices must have non-trivial global sections, got H⁰ = {}",
        h0
    );
    assert_eq!(h0, 1, "Identity restriction on edge: H⁰ = 1");

    // Verify the global section is the constant assignment (same value on both vertices)
    let basis = global_sections_basis(&sheaf);
    assert_eq!(basis.len(), 1, "Should have exactly 1 global section basis vector");
    assert_eq!(basis[0].len(), 2, "C⁰ dimension = 2");
    // The global section should be a multiple of (1, 1)
    let ratio = basis[0][1] / basis[0][0];
    assert!(
        (ratio - 1.0).abs() < 1e-10,
        "Global section should be constant: got ({}, {})",
        basis[0][0], basis[0][1]
    );
}

// 3. Disconnected sheaf: H⁰ = sum of component H⁰s

#[test]
fn disconnected_sheaf_h0_sum_of_components() {
    // Two isolated vertices with 1-dim stalks each
    let sheaf = Sheaf {
        num_vertices: 2,
        stalks: vec![VertexStalk(1), VertexStalk(1)],
        edges: vec![],
    };
    assert_eq!(dim_h0(&sheaf), 2, "Two isolated 1-dim vertices: H⁰ = 2");

    // Three isolated vertices with mixed dims
    let sheaf = Sheaf {
        num_vertices: 3,
        stalks: vec![VertexStalk(2), VertexStalk(3), VertexStalk(1)],
        edges: vec![],
    };
    assert_eq!(dim_h0(&sheaf), 6, "Isolated vertices (2+3+1 dims): H⁰ = 6");

    // One connected component (edge 0-1) + one isolated vertex
    let sheaf = Sheaf {
        num_vertices: 3,
        stalks: vec![VertexStalk(1), VertexStalk(1), VertexStalk(1)],
        edges: vec![EdgeRestriction {
            i: 0,
            j: 1,
            matrix: vec![vec![1.0]],
        }],
    };
    // Component 1 (vertices 0,1): H⁰ = 1. Component 2 (vertex 2): H⁰ = 1. Total = 2.
    assert_eq!(
        dim_h0(&sheaf), 2,
        "One connected pair + one isolated: H⁰ = 1 + 1 = 2"
    );
}

// 4. Sheaf Laplacian reduces to graph Laplacian for 1-dimensional stalks with identity restrictions

#[test]
fn sheaf_laplacian_equals_graph_laplacian() {
    // Test on a triangle (complete graph K3)
    let sheaf = Sheaf::constant(3, 1);
    let sl = sheaf_laplacian(&sheaf);
    let edges: Vec<(usize, usize)> = sheaf.edges.iter().map(|e| (e.i, e.j)).collect();
    let gl = graph_laplacian(3, &edges);

    for i in 0..3 {
        for j in 0..3 {
            let diff = (sl.data[i][j] - gl.data[i][j]).abs();
            assert!(
                diff < 1e-10,
                "K3: Sheaf Laplacian [{i},{j}] = {} != graph Laplacian {}",
                sl.data[i][j], gl.data[i][j]
            );
        }
    }

    // Test on path graph P4
    let sheaf = Sheaf::constant_path(4, 1);
    let sl = sheaf_laplacian(&sheaf);
    let edges: Vec<(usize, usize)> = sheaf.edges.iter().map(|e| (e.i, e.j)).collect();
    let gl = graph_laplacian(4, &edges);

    for i in 0..4 {
        for j in 0..4 {
            let diff = (sl.data[i][j] - gl.data[i][j]).abs();
            assert!(
                diff < 1e-10,
                "P4: Sheaf Laplacian [{i},{j}] = {} != graph Laplacian {}",
                sl.data[i][j], gl.data[i][j]
            );
        }
    }

    // Test on K5
    let sheaf = Sheaf::constant(5, 1);
    let sl = sheaf_laplacian(&sheaf);
    let edges: Vec<(usize, usize)> = sheaf.edges.iter().map(|e| (e.i, e.j)).collect();
    let gl = graph_laplacian(5, &edges);

    for i in 0..5 {
        for j in 0..5 {
            let diff = (sl.data[i][j] - gl.data[i][j]).abs();
            assert!(
                diff < 1e-10,
                "K5: Sheaf Laplacian [{i},{j}] = {} != graph Laplacian {}",
                sl.data[i][j], gl.data[i][j]
            );
        }
    }
}

// 5. Global consistency: 1.0 for constant sheaf on disconnected graph, less for connected

#[test]
fn global_consistency_disconnected_is_one() {
    // No edges → every assignment is a global section → consistency = 1.0
    let sheaf = Sheaf {
        num_vertices: 3,
        stalks: vec![VertexStalk(2), VertexStalk(3), VertexStalk(1)],
        edges: vec![],
    };
    let gc = global_consistency(&sheaf);
    assert!(
        (gc - 1.0).abs() < 1e-10,
        "Disconnected sheaf global consistency should be 1.0, got {}",
        gc
    );
}

#[test]
fn global_consistency_constant_path_less_than_one() {
    // Path graph on 4 vertices, stalk dim 2: H⁰ = 2, C⁰ = 8 → gc = 0.25
    let sheaf = Sheaf::constant_path(4, 2);
    let gc = global_consistency(&sheaf);
    assert!(
        gc < 1.0,
        "Connected sheaf global consistency < 1.0, got {}",
        gc
    );
    assert!(
        (gc - 0.25).abs() < 1e-10,
        "Constant path(4,2): gc = 2/8 = 0.25, got {}",
        gc
    );
}

// 6. Edge restriction matrix validation

#[test]
fn edge_restriction_dimensions_are_valid() {
    // Verify that restriction matrices have correct dimensions
    let sheaf = Sheaf::constant(3, 2);
    for edge in &sheaf.edges {
        let source_dim = sheaf.stalks[edge.i].0;
        let target_dim = sheaf.stalks[edge.j].0;
        assert_eq!(
            edge.matrix.len(),
            target_dim,
            "Edge ({}→{}): matrix rows {} != target stalk dim {}",
            edge.i, edge.j, edge.matrix.len(), target_dim
        );
        assert_eq!(
            edge.matrix[0].len(),
            source_dim,
            "Edge ({}→{}): matrix cols {} != source stalk dim {}",
            edge.i, edge.j, edge.matrix[0].len(), source_dim
        );
    }
}

#[test]
fn edge_restriction_identity_values() {
    // Constant sheaf restriction maps should be identity matrices
    let sheaf = Sheaf::constant(3, 3);
    for edge in &sheaf.edges {
        let k = sheaf.stalks[edge.i].0;
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (edge.matrix[i][j] - expected).abs() < 1e-10,
                    "Edge ({}→{}) matrix [{i}][{j}] = {} != {}",
                    edge.i, edge.j, edge.matrix[i][j], expected
                );
            }
        }
    }
}

#[test]
fn edge_restriction_projection_dimensions() {
    // 3-dim → 2-dim projection: matrix should be 2×3
    let proj: Vec<Vec<f64>> = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    let sheaf = Sheaf {
        num_vertices: 2,
        stalks: vec![VertexStalk(3), VertexStalk(2)],
        edges: vec![EdgeRestriction {
            i: 0,
            j: 1,
            matrix: proj.clone(),
        }],
    };
    assert_eq!(sheaf.edges[0].matrix.len(), 2, "2 rows for 2-dim target");
    assert_eq!(sheaf.edges[0].matrix[0].len(), 3, "3 cols for 3-dim source");
    assert_eq!(sheaf.dim_c0(), 5, "C⁰ = 3 + 2 = 5");
    assert_eq!(sheaf.dim_c1(), 2, "C¹ = 2 (one edge, 2-dim codomain)");
}
