use juniward::StcParams;
use juniward::stc::{StcError, stc_embed};

#[test]
fn embedding_fails_when_required_flip_has_no_finite_cost_candidate() {
    let cover_bits = [0, 0, 0, 0];
    let costs = [f64::INFINITY; 4];
    let message = [1];
    let params = StcParams::new(7);

    let err = stc_embed(&cover_bits, &costs, &message, &params).unwrap_err();

    assert!(matches!(err, StcError::EmbeddingFailed));
}

#[test]
fn embedding_can_leave_infinite_cost_group_unchanged_when_parity_already_matches() {
    let cover_bits = [1, 0, 1, 0];
    let costs = [f64::INFINITY; 4];
    let message = [0];
    let params = StcParams::new(7);

    let stego_bits = stc_embed(&cover_bits, &costs, &message, &params).expect("embedding failed");

    assert_eq!(stego_bits, cover_bits);
}
