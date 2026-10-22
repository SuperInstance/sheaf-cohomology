//! Integration tests for the Hodge decomposition API.
//!
//! The cross-cutting properties live here rather than in `src/hodge.rs`:
//! the orthogonal-projection property, the numerical-cliff regression, the
//! exact-field trap at fleet scale, and a randomised property sweep over
//! sheaves with singular and rank-deficient restriction maps.

use sheaf_cohomology::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn identity(k: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; k]; k];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn matvec(m: &Matrix, v: &[f64]) -> Vec<f64> {
    m.data
        .iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// A 0-cochain of the given length, with no obvious structure.
fn probe(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|i| (((i * 37 + seed * 11) % 19) as f64 - 9.0) * 0.31)
        .collect()
}

/// The twisted triangle from the crate's own test suite.
fn twisted_triangle() -> Sheaf {
    Sheaf {
        num_vertices: 3,
        stalks: vec![VertexStalk(2), VertexStalk(2), VertexStalk(2)],
        edges: vec![
            EdgeRestriction {
                i: 0,
                j: 1,
                matrix: identity(2),
            },
            EdgeRestriction {
                i: 1,
                j: 2,
                matrix: identity(2),
            },
            EdgeRestriction {
                i: 0,
                j: 2,
                matrix: vec![vec![1.0, 0.0], vec![0.0, -1.0]],
            },
        ],
    }
}

/// The projection sheaf: a 3-dimensional stalk restricted to a 2-dimensional one.
fn projection_sheaf() -> Sheaf {
    Sheaf {
        num_vertices: 2,
        stalks: vec![VertexStalk(3), VertexStalk(2)],
        edges: vec![EdgeRestriction {
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
        edges: vec![EdgeRestriction {
            i: 0,
            j: 1,
            matrix: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        }],
    }
}

/// Deterministic LCG, so every run of the sweep is reproducible.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize) % (hi - lo + 1)
    }
}

/// A random sheaf with random edges and random restriction matrices —
/// including all-zero (fully singular) and rank-deficient ones.
fn random_sheaf(rng: &mut Lcg) -> Sheaf {
    let n = rng.range(2, 6);
    let k = rng.range(1, 3);
    let m = rng.range(0, (n * (n - 1)) / 2);
    let stalks = vec![VertexStalk(k); n];
    let mut edges = Vec::new();
    for idx in 0..m {
        let i = idx % n;
        let j = (idx + 1) % n;
        if i == j {
            continue;
        }
        let mut mat = vec![vec![0.0; k]; k];
        for row in mat.iter_mut().take(k) {
            for cell in row.iter_mut().take(k) {
                let roll = rng.f64();
                *cell = if roll < 0.15 {
                    0.0 // singular
                } else if roll < 0.25 {
                    1.0 // rank-deficient
                } else {
                    (rng.f64() - 0.5) * 4.0 // entries in [-2, 2]
                };
            }
        }
        edges.push(EdgeRestriction { i, j, matrix: mat });
    }
    Sheaf {
        num_vertices: n,
        stalks,
        edges,
    }
}

// ---------------------------------------------------------------------------
// T4 — the required property
// ---------------------------------------------------------------------------

/// The harmonic representative is the **orthogonal projection onto `ker L`**.
///
/// The implementation under test computes `ker(δ)`; the basis of `ker L` used
/// here comes from the *shipped* `global_sections_basis` — the Gram-matrix
/// path `kernel_basis(L = δᵀδ)` — so this is not a tautology.
///
/// Kept to small sheaves: on a complete graph the Gram-matrix path is
/// numerically unsound from `n = 43` upward, and the test would then fail for
/// the wrong reason. The δ-based path is pinned separately by
/// `hodge_b0_is_correct_where_the_gram_matrix_path_is_not`.
#[test]
fn harmonic_is_the_orthogonal_projection_onto_ker_l() {
    let sheaf = Sheaf::constant(4, 2);
    let basis = hodge_basis(&sheaf);
    let s = vec![0.5, -1.25, 2.0, 0.75, -0.5, 1.0, 0.25, -2.0]; // dim C⁰ = 8

    let d = basis.decompose_c0(&s);

    // (P1) harmonic ∈ ker L
    let l = sheaf_laplacian(&sheaf);
    let lh = matvec(&l, &d.harmonic);
    assert!(norm(&lh) < 1e-10, "harmonic must lie in ker L");

    // (P2) the residual is orthogonal to ker L
    // (P3) harmonic agrees with s on ker L
    let ker_l = global_sections_basis(&sheaf); // INDEPENDENT path
    assert!(!ker_l.is_empty());
    let residual: Vec<f64> = s.iter().zip(d.harmonic.iter()).map(|(a, b)| a - b).collect();
    for u in &ker_l {
        assert!(
            dot(&residual, u).abs() < 1e-10,
            "residual must be orthogonal to ker L"
        );
        assert!((dot(&d.harmonic, u) - dot(&s, u)).abs() < 1e-10);
    }
}

/// The same property, on a spread of structured sheaves.
#[test]
fn harmonic_is_the_orthogonal_projection_onto_ker_l_on_structured_sheaves() {
    let cases = [
        ("constant(3, 2)", Sheaf::constant(3, 2)),
        ("constant(5, 1)", Sheaf::constant(5, 1)),
        ("constant(6, 2)", Sheaf::constant(6, 2)),
        ("constant_path(4, 2)", Sheaf::constant_path(4, 2)),
        ("constant_path(6, 3)", Sheaf::constant_path(6, 3)),
        ("twisted triangle", twisted_triangle()),
        ("projection sheaf", projection_sheaf()),
        ("zero restriction", zero_restriction_sheaf()),
        (
            "disconnected",
            Sheaf {
                num_vertices: 3,
                stalks: vec![VertexStalk(1), VertexStalk(1), VertexStalk(1)],
                edges: vec![],
            },
        ),
    ];

    for (name, sheaf) in &cases {
        let basis = hodge_basis(sheaf);
        let s = probe(basis.dim_c0, 3);

        let d = basis.decompose_c0(&s);

        // (P1) harmonic ∈ ker L
        let l = sheaf_laplacian(sheaf);
        let lh = matvec(&l, &d.harmonic);
        assert!(
            norm(&lh) < 1e-9 * (1.0 + norm(&d.harmonic)),
            "{name}: harmonic must lie in ker L"
        );

        // (P2) residual ⟂ ker L, (P3) harmonic agrees with s on ker L
        let ker_l = global_sections_basis(sheaf); // INDEPENDENT path
        let residual: Vec<f64> = s.iter().zip(d.harmonic.iter()).map(|(a, b)| a - b).collect();
        for u in &ker_l {
            assert!(
                dot(&residual, u).abs() < 1e-9 * (1.0 + norm(&residual) * norm(u)),
                "{name}: residual must be orthogonal to ker L"
            );
            assert!(
                (dot(&d.harmonic, u) - dot(&s, u)).abs() < 1e-9 * (1.0 + norm(&s) * norm(u)),
                "{name}: harmonic must agree with s on ker L"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// T2 — agreement with the shipped cohomological dimensions
// ---------------------------------------------------------------------------

#[test]
fn hodge_b1_matches_the_shipped_dim_h1() {
    for (name, sheaf) in [
        ("constant(4, 2)", Sheaf::constant(4, 2)),
        ("constant(6, 1)", Sheaf::constant(6, 1)),
        ("constant_path(5, 3)", Sheaf::constant_path(5, 3)),
        ("twisted triangle", twisted_triangle()),
        ("projection sheaf", projection_sheaf()),
        ("zero restriction", zero_restriction_sheaf()),
    ] {
        let basis = hodge_basis(&sheaf);
        assert_eq!(basis.b1, dim_h1(&sheaf), "{name}: b1 must match dim_h1");
    }
}

// ---------------------------------------------------------------------------
// T17 — the numerical cliff
// ---------------------------------------------------------------------------

/// `dim_h0 = nullity(L)` returns 0 (instead of k) from n = 43 upward on a
/// complete graph, because `L = δᵀδ` squares δ's condition number and
/// `Matrix::rank` uses an ABSOLUTE pivot tolerance of 1e-12.
///
/// The Hodge path computes `b0 = dim_c0 − rank(δ)`, from the coboundary
/// itself, and is correct at every size.
#[test]
fn hodge_b0_is_correct_where_the_gram_matrix_path_is_not() {
    for n in [43usize, 50] {
        for k in [1usize, 4] {
            let sheaf = Sheaf::constant(n, k);
            let basis = hodge_basis(&sheaf);
            assert_eq!(basis.b0, k, "b0 must be k at n={n}, k={k}");
        }
    }
}

/// The same regression at the larger cliff sizes.
///
/// This one is slow in a debug build (the dense RREF of δ is `Θ(E·n²·k³)`),
/// so it is ignored by default. Run it with:
///
/// ```text
/// cargo test --release -- --ignored
/// ```
#[test]
#[ignore = "slow in debug; run with `cargo test --release -- --ignored`"]
fn hodge_b0_is_correct_at_the_larger_cliff_sizes() {
    for n in [52usize, 54, 75, 100] {
        for k in [1usize, 4] {
            let sheaf = Sheaf::constant(n, k);
            let basis = hodge_basis(&sheaf);
            assert_eq!(basis.b0, k, "b0 must be k at n={n}, k={k}");
        }
    }
}

// ---------------------------------------------------------------------------
// T18 — the exact-field trap, at fleet scale
// ---------------------------------------------------------------------------

/// `P_H¹(δs) ≡ 0` for every `s`: `δs` is in `im(δ)` by construction, and
/// `im(δ) ⊥ ker(δ*)`.
///
/// A monitor that decomposes `δs` and reads off the H¹ component reports a
/// healthy mesh forever. The trap is pinned here so that it is a documented
/// behaviour rather than something a user discovers.
#[test]
fn exact_field_trap_at_fleet_scale() {
    // The tells fleet is 6–30 tells with k ≤ 4 (SHEAF-NOTES.md §2).
    for (n, k) in [(10usize, 1usize), (20, 2), (30, 4)] {
        let sheaf = Sheaf::constant(n, k);
        let basis = hodge_basis(&sheaf);
        let s = probe(basis.dim_c0, 4);
        let delta = coboundary(&sheaf).to_dense();
        let ds = matvec(&delta, &s);
        let d = basis.decompose_c1(&ds);
        assert!(
            norm(&d.harmonic) < 1e-9,
            "P_H¹(δs) must vanish at n={n}, k={k} (the exact-field trap)"
        );
        // The exact part, on the other hand, carries the whole field.
        assert!(
            (norm(&d.exact) - norm(&ds)).abs() < 1e-9 * (1.0 + norm(&ds)),
            "δs must be entirely exact"
        );
    }
}

// ---------------------------------------------------------------------------
// The randomised property sweep
// ---------------------------------------------------------------------------

/// Every structural property, on 300 randomised sheaves with singular and
/// rank-deficient restriction maps.
///
/// This is the "does the *structure* hold" kind of test, not the "does the
/// number come out right" kind.
#[test]
fn randomised_property_sweep() {
    let mut rng = Lcg(0x5EED_1234_ABCD_0001);
    let mut passed = 0usize;
    for trial in 0..300 {
        let sheaf = random_sheaf(&mut rng);
        if check_sheaf(&sheaf, trial) {
            passed += 1;
        }
    }
    assert_eq!(passed, 300, "every randomised sheaf must satisfy every property");
}

/// Every structural property of the Hodge decomposition on one sheaf.
fn check_sheaf(sheaf: &Sheaf, trial: usize) -> bool {
    let basis = hodge_basis(sheaf);
    let delta = coboundary(sheaf).to_dense();
    let delta_star = delta.transpose();

    // (1) The dimensions agree with the shipped path.
    assert_eq!(
        basis.b1,
        dim_h1(sheaf),
        "trial {trial}: b1 must match dim_h1"
    );
    assert_eq!(
        basis.b0,
        basis.ker_delta.len(),
        "trial {trial}: b0 must be the size of the ker(δ) basis"
    );

    // (2) The C⁰ decomposition.
    let s = probe(basis.dim_c0, trial);
    let d = basis.decompose_c0(&s);

    // (2a) harmonic ∈ ker(δ).
    let dh = matvec(&delta, &d.harmonic);
    assert!(
        norm(&dh) < 1e-8 * (1.0 + norm(&d.harmonic)),
        "trial {trial}: δ · harmonic must be zero"
    );

    // (2b) harmonic + coexact == s.
    for ((h, co), x) in d.harmonic.iter().zip(d.coexact.iter()).zip(s.iter()) {
        assert!(
            (h + co - x).abs() < 1e-10,
            "trial {trial}: harmonic + coexact must equal s"
        );
    }

    // (2c) harmonic ⟂ coexact.
    assert!(
        dot(&d.harmonic, &d.coexact).abs() < 1e-8 * (1.0 + norm(&d.harmonic) * norm(&d.coexact)),
        "trial {trial}: harmonic must be orthogonal to coexact"
    );

    // (2d) coexact ∈ im(δ*) = (ker δ)⊥.
    for u in &basis.ker_delta {
        assert!(
            dot(&d.coexact, u).abs() < 1e-8 * (1.0 + norm(&d.coexact) * norm(u)),
            "trial {trial}: coexact must be orthogonal to ker(δ)"
        );
    }

    // (3) The C¹ decomposition, against an INDEPENDENT basis of ker(δ*).
    let c = probe(basis.dim_c1, trial + 1000);
    let d1 = basis.decompose_c1(&c);

    // (3a) harmonic ∈ ker(δ*), checked against the shipped kernel_basis of δᵀ.
    let ker_ds = delta_star.kernel_basis();
    assert_eq!(
        ker_ds.len(),
        basis.b1,
        "trial {trial}: an independent ker(δ*) must have dimension b1"
    );
    let residual: Vec<f64> = c.iter().zip(d1.harmonic.iter()).map(|(a, b)| a - b).collect();
    for u in &ker_ds {
        assert!(
            dot(&residual, u).abs() < 1e-7 * (1.0 + norm(&residual) * norm(u)),
            "trial {trial}: the exact part must be orthogonal to ker(δ*)"
        );
    }

    // (3b) harmonic + exact == c.
    for ((h, ex), x) in d1.harmonic.iter().zip(d1.exact.iter()).zip(c.iter()) {
        assert!(
            (h + ex - x).abs() < 1e-9,
            "trial {trial}: harmonic + exact must equal c"
        );
    }

    // (3c) harmonic ⟂ exact.
    assert!(
        dot(&d1.harmonic, &d1.exact).abs() < 1e-8 * (1.0 + norm(&d1.harmonic) * norm(&d1.exact)),
        "trial {trial}: harmonic must be orthogonal to exact"
    );

    // (4) The exact-field trap: P_H¹(δs) ≡ 0.
    let ds = matvec(&delta, &s);
    let d_trap = basis.decompose_c1(&ds);
    assert!(
        norm(&d_trap.harmonic) < 1e-8,
        "trial {trial}: P_H¹(δs) must vanish (the exact-field trap)"
    );

    // (5) The free functions and the cached methods must agree.
    let free = hodge_decompose_c0(sheaf, &s);
    for i in 0..s.len() {
        assert!(
            (free.harmonic[i] - d.harmonic[i]).abs() < 1e-12,
            "trial {trial}: hodge_decompose_c0 must match HodgeBasis::decompose_c0"
        );
    }
    let free1 = hodge_decompose_c1(sheaf, &c);
    for i in 0..c.len() {
        assert!(
            (free1.harmonic[i] - d1.harmonic[i]).abs() < 1e-12,
            "trial {trial}: hodge_decompose_c1 must match HodgeBasis::decompose_c1"
        );
    }

    true
}
