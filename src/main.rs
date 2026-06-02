use sheaf_cohomology::*;

fn main() {
    println!("=== Cellular Sheaf Cohomology ===\n");

    // Triangle complex
    println!("--- Triangle Complex ---");
    let tri = triangle_complex();
    println!("Cells: {}V, {}E, {}F", tri.num_cells(0), tri.num_cells(1), tri.num_cells(2));
    println!("Euler characteristic: {}", tri.euler_characteristic());
    println!("Betti numbers: {:?}", tri.betti_numbers());
    println!("Connected: {}", tri.is_connected());

    let sheaf = constant_sheaf(&tri);
    let cohom = SheafCohomology::compute(&sheaf);
    println!("H⁰ = {}, H¹ = {}", cohom.h0_dimension, cohom.h1_dimension);
    let lap = cohom.sheaf_laplacian_0();
    println!("Eigenvalues: {:?}", lap.eigenvalues());
    println!("Spectral gap: {:.4}", lap.spectral_gap());
    println!("Functoriality: {}", sheaf.verify_functoriality());

    // Circle
    println!("\n--- Circle S¹ (5 vertices) ---");
    let circ = circle_complex(5);
    println!("Euler: {}, Betti: {:?}", circ.euler_characteristic(), circ.betti_numbers());
    let sheaf = constant_sheaf(&circ);
    let cohom = SheafCohomology::compute(&sheaf);
    println!("H⁰ = {}, H¹ = {}", cohom.h0_dimension, cohom.h1_dimension);

    // Tetrahedron
    println!("\n--- Tetrahedron (S²) ---");
    let tet = tetrahedron_complex();
    println!("Cells: {}V, {}E, {}F", tet.num_cells(0), tet.num_cells(1), tet.num_cells(2));
    println!("Euler: {}, Betti: {:?}", tet.euler_characteristic(), tet.betti_numbers());
    let sheaf = constant_sheaf(&tet);
    let cohom = SheafCohomology::compute(&sheaf);
    println!("H⁰ = {}, H¹ = {}", cohom.h0_dimension, cohom.h1_dimension);
    let lap = cohom.sheaf_laplacian_0();
    println!("Eigenvalues: {:?}", lap.eigenvalues());

    // Orientation sheaf on S¹
    println!("\n--- Orientation Sheaf on S¹ ---");
    let circ = circle_complex(5);
    let sheaf = orientation_sheaf(&circ);
    println!("Functoriality: {}", sheaf.verify_functoriality());
    let cohom = SheafCohomology::compute(&sheaf);
    println!("H⁰ = {}, H¹ = {}", cohom.h0_dimension, cohom.h1_dimension);
}
