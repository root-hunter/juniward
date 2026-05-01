use juniward::{
    EmbedConfig, JuniwardError, StcParams, compute_costs, embed, embed_with_params, extract,
    extract_with_params,
};

const SAMPLE_JPEG: &[u8] = include_bytes!("../../tests/images/sample.jpg");

#[test]
fn compute_costs_returns_sane_dct_layout() {
    let (costs, stats) = compute_costs(SAMPLE_JPEG, 1e-10);

    assert!(!costs.is_empty());
    assert_eq!(costs.len() % 64, 0);
    assert!(costs.iter().all(|cost| cost.is_finite()));
    assert!(stats.min.is_finite());
    assert!(stats.max.is_finite());
    assert!(stats.mean.is_finite());
    assert!(stats.min >= 0.0);
    assert!(stats.min <= stats.mean);
    assert!(stats.mean <= stats.max);
}

#[test]
fn embed_and_extract_roundtrip_with_default_params() {
    let message = b"jpeg roundtrip";

    let stego = embed(SAMPLE_JPEG, message, EmbedConfig::default()).expect("embedding failed");
    let recovered = extract(&stego, message.len()).expect("extraction failed");

    assert_eq!(recovered, message);
    assert_ne!(stego, SAMPLE_JPEG);
}

#[test]
fn embed_and_extract_roundtrip_with_keyed_params() {
    let message = b"keyed jpeg roundtrip";
    let params = StcParams::from_key(b"test-password", 7, 16);

    let stego = embed_with_params(SAMPLE_JPEG, message, EmbedConfig::default(), &params)
        .expect("embedding failed");
    let recovered = extract_with_params(&stego, message.len(), &params).expect("extraction failed");

    assert_eq!(recovered, message);
}

#[test]
fn payload_too_large_reports_requested_and_available_bits() {
    let message = vec![0u8; 128 * 1024];

    let err = embed(SAMPLE_JPEG, &message, EmbedConfig::default()).unwrap_err();

    match err {
        JuniwardError::PayloadTooLarge {
            payload_bits,
            max_bits,
        } => {
            assert_eq!(payload_bits, message.len() * 8);
            assert!(max_bits > 0);
            assert!(max_bits < payload_bits);
        }
        other => panic!("expected payload-too-large error, got {other:?}"),
    }
}
