//! # Sheaf Cohomology Tutorial
//!
//! Progressive lessons covering cellular sheaves, coboundary operators,
//! sheaf Laplacians, cohomology groups, and consistency measures.
//!
//! Run with: `cargo run --example tutorial`

use sheaf_cohomology::{
    coboundary, coboundary_adjoint, dim_h0, dim_h1, euler_characteristic,
    global_consistency, global_sections_basis, graph_laplacian, local_consistency,
    sheaf_laplacian, EdgeRestriction, Matrix, Sheaf, VertexStalk,
};

// ── Lesson 1: Cellular Sheaves on Graphs ─────────────────────────────────────

fn lesson_1_what_is_a_sheaf() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 1: What is a Cellular Sheaf?");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("A cellular sheaf on a graph assigns:");
    println!("  • A vector space (stalk) to each vertex");
    println!("  • A linear map (restriction) to each edge");
    println!();

    // Simple sheaf: 2 vertices, 1-dim stalks, identity restriction
    let sheaf = Sheaf {
        num_vertices: 2,
        stalks: vec![VertexStalk(1), VertexStalk(1)],
        edges: vec![EdgeRestriction {
            i: 0,
            j: 1,
            matrix: vec![vec![1.0]],
        }],
    };

    println!("  Example: Edge with 1-dimensional stalks");
    println!("    Vertices: 2, each with stalk dimension 1");
    println!("    Edge {{0,1}} restriction: identity (1×1 matrix [[1]])");
    println!("    dim(C⁰) = {}, dim(C¹) = {}", sheaf.dim_c0(), sheaf.dim_c1());
    println!();

    // Disconnected sheaf
    let disconnected = Sheaf {
        num_vertices: 3,
        stalks: vec![VertexStalk(1), VertexStalk(1), VertexStalk(1)],
        edges: vec![],
    };
    println!("  Disconnected sheaf (3 isolated vertices):");
    println!("    dim(C⁰) = {}, dim(C¹) = {}", disconnected.dim_c0(), disconnected.dim_c1());
    println!();
}

// ── Lesson 2: Constant Sheaves ───────────────────────────────────────────────

fn lesson_2_constant_sheaves() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 2: Constant Sheaves");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("A constant sheaf assigns the same stalk to every vertex and the identity");
    println!("map to every edge. It's the simplest nontrivial sheaf.");
    println!();

    // Constant sheaf on complete graph K₃ with stalk dim 2
    let sheaf = Sheaf::constant(3, 2);
    println!("  Constant sheaf on K₃ (complete graph, 3 vertices, stalk dim 2):");
    println!("    Vertices: {}", sheaf.num_vertices);
    println!("    Edges: {}", sheaf.edges.len());
    println!("    dim(C⁰) = {} (3 vertices × 2-dim stalks)", sheaf.dim_c0());
    println!("    dim(C¹) = {} (3 edges × 2-dim codomain)", sheaf.dim_c1());
    println!();

    // Constant sheaf on a path graph (tree)
    let path = Sheaf::constant_path(5, 1);
    println!("  Constant sheaf on P₅ (path graph, 5 vertices, stalk dim 1):");
    println!("    Vertices: {}", path.num_vertices);
    println!("    Edges: {} (path has n-1 edges)", path.edges.len());
    println!("    dim(C⁰) = {}, dim(C¹) = {}", path.dim_c0(), path.dim_c1());
    println!();
}

// ── Lesson 3: The Coboundary Operator ────────────────────────────────────────

fn lesson_3_coboundary() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 3: The Coboundary Operator δ: C⁰ → C¹");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("The coboundary δ measures how much a 0-cochain (assignment of vectors to");
    println!("vertices) violates the restriction maps. For edge {{i,j}}:");
    println!("  (δs)_{{i,j}} = F·sᵢ − sⱼ");
    println!();

    let sheaf = Sheaf::constant(3, 1);
    let delta = coboundary(&sheaf);
    println!("  Coboundary for constant sheaf on K₃ (stalk dim 1):");
    println!("    Matrix shape: {} rows × {} cols", delta.rows, delta.cols);
    println!();

    let dense = delta.to_dense();
    println!("    δ (as dense matrix):");
    for row in &dense.data {
        print!("      [");
        for (j, v) in row.iter().enumerate() {
            if j > 0 { print!(", "); }
            print!("{:5.1}", v);
        }
        println!("]");
    }
    println!();

    // Adjoint
    let delta_star = coboundary_adjoint(&sheaf);
    println!("    δ* (adjoint) shape: {} × {}", delta_star.rows, delta_star.cols);
    println!();
}

// ── Lesson 4: The Sheaf Laplacian ────────────────────────────────────────────

fn lesson_4_sheaf_laplacian() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 4: The Sheaf Laplacian L = δ*δ");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("The sheaf Laplacian L = δ*δ measures local inconsistency.");
    println!("For 1-dimensional stalks with identity restrictions, it reduces to");
    println!("the ordinary graph Laplacian.");
    println!();

    let sheaf = Sheaf::constant(3, 1);
    let sl = sheaf_laplacian(&sheaf);
    println!("  Sheaf Laplacian for constant sheaf on K₃:");
    println!("{}", sl);
    println!();

    // Compare with graph Laplacian
    let edges: Vec<(usize, usize)> = sheaf.edges.iter().map(|e| (e.i, e.j)).collect();
    let gl = graph_laplacian(3, &edges);
    println!("  Graph Laplacian (should be identical):");
    println!("{}", gl);
    println!();

    // Verify they match
    let matches = sl.data.iter().zip(&gl.data).all(|(r1, r2)| {
        r1.iter().zip(r2).all(|(a, b)| (a - b).abs() < 1e-10)
    });
    println!("  Sheaf Laplacian == Graph Laplacian? {}", matches);
    println!();
}

// ── Lesson 5: Cohomology Groups H⁰ and H¹ ────────────────────────────────────

fn lesson_5_cohomology() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 5: Cohomology Groups H⁰ and H¹");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("H⁰ = ker(L) = global sections (assignments that agree everywhere).");
    println!("H¹ = coker(δ) = obstructions to extending local data globally.");
    println!();

    // Constant sheaf on tree: H⁰ = stalk_dim, H¹ = 0
    let tree = Sheaf::constant_path(4, 2);
    let h0_tree = dim_h0(&tree);
    let h1_tree = dim_h1(&tree);
    println!("  Constant sheaf on path P₄ (stalk dim 2):");
    println!("    dim(H⁰) = {} (all vertices agree in both coordinates)", h0_tree);
    println!("    dim(H¹) = {} (trees have no obstructions)", h1_tree);
    println!();

    // Constant sheaf on triangle (has a cycle): H⁰ = 1, H¹ = 1 for dim-1 stalks
    let triangle = Sheaf {
        num_vertices: 3,
        stalks: vec![VertexStalk(1), VertexStalk(1), VertexStalk(1)],
        edges: vec![
            EdgeRestriction { i: 0, j: 1, matrix: vec![vec![1.0]] },
            EdgeRestriction { i: 1, j: 2, matrix: vec![vec![1.0]] },
            EdgeRestriction { i: 0, j: 2, matrix: vec![vec![1.0]] },
        ],
    };
    let h0_tri = dim_h0(&triangle);
    let h1_tri = dim_h1(&triangle);
    println!("  Constant sheaf on triangle C₃ (stalk dim 1):");
    println!("    dim(H⁰) = {} (all vertices agree)", h0_tri);
    println!("    dim(H¹) = {} (the cycle creates one obstruction class)", h1_tri);
    println!();

    // Disconnected sheaf
    let disc = Sheaf {
        num_vertices: 3,
        stalks: vec![VertexStalk(2), VertexStalk(2), VertexStalk(2)],
        edges: vec![],
    };
    println!("  Disconnected sheaf (3 isolated vertices, stalk dim 2):");
    println!("    dim(H⁰) = {} (= Σ stalk dims, maximum freedom)", dim_h0(&disc));
    println!("    dim(H¹) = {} (no edges = no obstructions)", dim_h1(&disc));
    println!();
}

// ── Lesson 6: Global Sections ────────────────────────────────────────────────

fn lesson_6_global_sections() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 6: Global Sections (Kernel of L)");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("Global sections are assignments of vectors to vertices that satisfy ALL");
    println!("restriction maps simultaneously. They form the kernel of the sheaf Laplacian.");
    println!();

    // Constant sheaf on K₃ with stalk dim 2
    let sheaf = Sheaf::constant(3, 2);
    let basis = global_sections_basis(&sheaf);
    println!("  Constant sheaf on K₃ (stalk dim 2):");
    println!("    dim(H⁰) = {} global section(s)", basis.len());
    println!();

    // Display the basis vectors (laid out as 6-dimensional C⁰ vectors)
    println!("    Global section basis (each vector spans C⁰ = R⁶):");
    for (i, v) in basis.iter().enumerate() {
        print!("      basis {}: [", i);
        for (j, val) in v.iter().enumerate() {
            if j > 0 { print!(", "); }
            print!("{:.1}", val);
        }
        println!("]");
        // Interpret: first 2 coords = vertex 0, next 2 = vertex 1, last 2 = vertex 2
        println!("        → vertex 0: [{:.1}, {:.1}], vertex 1: [{:.1}, {:.1}], vertex 2: [{:.1}, {:.1}]",
                 v[0], v[1], v[2], v[3], v[4], v[5]);
    }
    println!();
}

// ── Lesson 7: Consistency Measures ────────────────────────────────────────────

fn lesson_7_consistency() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 7: Local and Global Consistency");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("Local consistency: how well a vertex agrees with its neighbors (0, 1].");
    println!("Global consistency: H⁰/C⁰ ratio measuring overall sheaf coherence.");
    println!();

    // Perfect consistency: disconnected
    let disc = Sheaf {
        num_vertices: 3,
        stalks: vec![VertexStalk(1), VertexStalk(1), VertexStalk(1)],
        edges: vec![],
    };
    println!("  Disconnected sheaf (no edges):");
    println!("    Global consistency: {:.2} (maximal)", global_consistency(&disc));
    for v in 0..3 {
        println!("    Local consistency at vertex {}: {:.2}", v, local_consistency(&disc, v));
    }
    println!();

    // Connected graph
    let path = Sheaf::constant_path(3, 2);
    println!("  Constant sheaf on path P₃ (stalk dim 2):");
    println!("    Global consistency: {:.4}", global_consistency(&path));
    println!("    H⁰ = {}, C⁰ = {}, ratio = {:.4}",
             dim_h0(&path), path.dim_c0(),
             dim_h0(&path) as f64 / path.dim_c0() as f64);
    println!();

    // Twisted sheaf: lower consistency
    let twisted = Sheaf {
        num_vertices: 3,
        stalks: vec![VertexStalk(2), VertexStalk(2), VertexStalk(2)],
        edges: vec![
            EdgeRestriction { i: 0, j: 1, matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]] },
            EdgeRestriction { i: 1, j: 2, matrix: vec![vec![1.0, 0.0], vec![0.0, 1.0]] },
            EdgeRestriction { i: 0, j: 2, matrix: vec![vec![1.0, 0.0], vec![0.0, -1.0]] },
        ],
    };
    println!("  Twisted sheaf (one edge flips a coordinate):");
    println!("    H⁰ = {} (twisted) vs H⁰ = {} (constant on same graph)",
             dim_h0(&twisted), dim_h0(&Sheaf::constant(3, 2)));
    println!("    Global consistency: {:.4} (twisted) vs {:.4} (constant)",
             global_consistency(&twisted), global_consistency(&Sheaf::constant(3, 2)));
    println!();
}

// ── Lesson 8: Euler Characteristic & Matrix Algebra ──────────────────────────

fn lesson_8_euler_and_matrix() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 8: Euler Characteristic & Matrix Operations");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("The Euler characteristic χ = dim(H⁰) − dim(H¹) is a topological invariant.");
    println!();

    // Trees
    for n in [2, 3, 4, 5] {
        let path = Sheaf::constant_path(n, 1);
        let chi = euler_characteristic(&path);
        println!("  Path P_{} (tree): χ = {} (H⁰={}, H¹={})",
                 n, chi, dim_h0(&path), dim_h1(&path));
    }
    println!();

    // Cycle
    let cycle = Sheaf {
        num_vertices: 3,
        stalks: vec![VertexStalk(1), VertexStalk(1), VertexStalk(1)],
        edges: vec![
            EdgeRestriction { i: 0, j: 1, matrix: vec![vec![1.0]] },
            EdgeRestriction { i: 1, j: 2, matrix: vec![vec![1.0]] },
            EdgeRestriction { i: 0, j: 2, matrix: vec![vec![1.0]] },
        ],
    };
    println!("  Triangle C₃: χ = {} (H⁰={}, H¹={})",
             euler_characteristic(&cycle), dim_h0(&cycle), dim_h1(&cycle));
    println!();

    // Matrix operations
    println!("  Matrix rank & nullity example:");
    let mut m = Matrix::zero(3, 4);
    m.set(0, 0, 1.0); m.set(0, 1, 2.0); m.set(0, 2, 0.0); m.set(0, 3, 1.0);
    m.set(1, 0, 0.0); m.set(1, 1, 0.0); m.set(1, 2, 1.0); m.set(1, 3, 3.0);
    m.set(2, 0, 2.0); m.set(2, 1, 4.0); m.set(2, 2, 0.0); m.set(2, 3, 2.0);
    println!("    Matrix (3×4):");
    for row in &m.data {
        println!("      {:?}", row);
    }
    println!("    Rank: {}, Nullity: {}", m.rank(), m.nullity());
    println!();

    let kernel = m.kernel_basis();
    println!("    Kernel basis vectors:");
    for (i, v) in kernel.iter().enumerate() {
        println!("      {}: {:?}", i, v);
    }
    println!();
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("╔═══════════════════════════════════════════════════╗");
    println!("║   Sheaf Cohomology Tutorial                      ║");
    println!("║   Cellular Sheaves and Cohomology on Graphs      ║");
    println!("╚═══════════════════════════════════════════════════╝");
    println!();

    lesson_1_what_is_a_sheaf();
    lesson_2_constant_sheaves();
    lesson_3_coboundary();
    lesson_4_sheaf_laplacian();
    lesson_5_cohomology();
    lesson_6_global_sections();
    lesson_7_consistency();
    lesson_8_euler_and_matrix();

    println!("═══════════════════════════════════════════════════");
    println!("  Tutorial complete! Key takeaways:");
    println!("    • Sheaves assign data + compatibility to graph structures");
    println!("    • The coboundary δ measures restriction violations");
    println!("    • H⁰ = global agreement, H¹ = obstructions from cycles");
    println!("    • Sheaf Laplacian generalizes the graph Laplacian");
    println!("═══════════════════════════════════════════════════");
}
