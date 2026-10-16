//! The demo generator must draw exactly what ``np.random.default_rng`` draws.

use hkfactor::nprandom::Generator;
use sha2::{Digest, Sha256};

#[test]
fn matches_numpy_default_rng() {
    let mut rng = Generator::default_rng(101);
    let draws = rng.normal(0.0001, 0.008, 200_000);
    let ints = rng.integers(200_000, 500_000, 7);
    // Reference values from NumPy 2.x: default_rng(101).normal(...), .integers(...)
    assert_eq!(draws[0], -0.006221219999704116);
    assert_eq!(draws[1], -0.016177003854654984);
    assert_eq!(draws[199_999], -0.011591825615350578);
    assert_eq!(
        ints,
        vec![311007, 428751, 450135, 309148, 305595, 282117, 256346]
    );
    let mut bytes = Vec::new();
    for v in &draws {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(
        hex::encode(Sha256::digest(&bytes)),
        "c09565ff77b4bf6898d34dc8e4559a26b14c5302cf106edeb523d175174e23fd"
    );
}
