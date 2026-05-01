use juniward_rs::{embed, extract, compute_costs, EmbedConfig};
use std::fs;

fn main() {
    println!("=== J-UNIWARD + STC Steganography ===\n");

    // 1. Read cover image
    let data = fs::read("image.jpg").expect("Place image.jpg in the current directory");
    println!("Cover JPEG read: {} bytes", data.len());

    // 2. Show cost statistics
    println!("\n[1/4] Computing J-UNIWARD costs...");
    let (_, stats) = compute_costs(&data, 1e-10);
    println!(
        "  Cost min: {:.4}, max: {:.4}, mean: {:.4}",
        stats.min, stats.max, stats.mean
    );

    // 3. Prepare message
    let message = b"Secret message hidden in the JPEG using J-UNIWARD and STC!";
    println!("\n[2/4] Message: \"{}\"", std::str::from_utf8(message).unwrap());
    println!("  Length: {} bytes = {} bits", message.len(), message.len() * 8);

    // 4. Embed
    println!("\n[3/4] Embedding...");
    let cfg = EmbedConfig::default();
    let stego_data = match embed(&data, message, cfg) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Embedding failed: {e}");
            return;
        }
    };

    // 5. Write stego JPEG
    println!("\n[4/4] Writing stego.jpg...");
    fs::write("stego.jpg", &stego_data).expect("Could not write stego.jpg");
    println!(
        "  stego.jpg written: {} bytes (original: {} bytes)",
        stego_data.len(),
        data.len()
    );

    // 6. Verify: extract message from stego
    println!("\n=== DECODING VERIFICATION ===");
    match extract(&stego_data, message.len()) {
        Ok(recovered) => {
            let recovered_text = String::from_utf8_lossy(&recovered);
            println!("Recovered message: \"{recovered_text}\"");

            let bit_errors: usize = message
                .iter()
                .zip(recovered.iter())
                .flat_map(|(a, b)| (0..8).map(move |i| ((a >> i) & 1) != ((b >> i) & 1)))
                .filter(|&x| x)
                .count();
            println!("Bit errors: {bit_errors} / {}", message.len() * 8);

            if bit_errors == 0 {
                println!("\n✓ Embedding and decoding completed successfully!");
            } else {
                println!("\n✗ Warning: {bit_errors} bit errors in recovered message");
            }
        }
        Err(e) => eprintln!("Extraction failed: {e}"),
    }
}
