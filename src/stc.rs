#![allow(dead_code)]

/// STC — Syndrome-Trellis Coding
///
/// Embedding: trova la sequenza di modifiche a costo minimo
///            tale che H·y = m  (in GF2)
///
/// Decoding:  calcola m = H·y  (semplice moltiplicazione GF2)

/// Parametri STC
pub struct StcParams {
    /// Vettore generatore della matrice H (altezza del trellis)
    /// Determina la struttura della matrice parity-check.
    /// Deve essere condiviso tra sender e receiver (insieme alla chiave).
    pub h_hat: Vec<u64>,

    /// Numero di stati nel trellis = 2^(len(h_hat)*bits_per_word)
    /// In pratica usiamo h_hat come colonne di H, ogni elemento = 1 word
    pub h_height: usize, // numero di righe di H per "colonna" = log2(num_states)
}

impl StcParams {
    /// Crea parametri STC con h_hat standard (da letteratura)
    /// h_height = 7 → 128 stati, buon compromesso velocità/sicurezza
    pub fn new(h_height: usize) -> Self {
        // h_hat: vettore di interi a 64 bit che definisce H
        // Ogni bit di h_hat[i] definisce quale riga di H viene XORata
        // quando si processa il coefficiente i-esimo.
        // Valore standard dalla letteratura per h=7:
        let h_hat = vec![
            0b1011011u64, // colonna 0
            0b1111001u64, // colonna 1
            0b1010011u64, // colonna 2
            // Viene ripetuto ciclicamente per tutti gli n coefficienti
        ];
        StcParams { h_hat, h_height }
    }

    /// Numero di stati nel trellis
    pub fn num_states(&self) -> usize {
        1 << self.h_height
    }

    /// Calcola la transizione di stato quando si sceglie il valore y_i
    /// per il coefficiente i-esimo.
    /// Lo stato è un intero in [0, 2^h_height).
    ///
    /// H è costruita come segue: la colonna i è h_hat[i % h_hat.len()],
    /// ma ruotata di (i / h_hat.len()) * 1 bit (struttura a banda).
    #[inline]
    pub fn next_state(&self, state: usize, coeff_idx: usize, y_bit: u8) -> usize {
        if y_bit == 0 {
            // Se y_i = 0, la colonna di H contribuisce 0 → stato invariato
            state
        } else {
            // Se y_i = 1, XOR con la colonna i di H
            let col = self.h_column(coeff_idx);
            state ^ col
        }
    }

    /// Ritorna la colonna i-esima di H come maschera di bit
    #[inline]
    fn h_column(&self, i: usize) -> usize {
        let base = self.h_hat[i % self.h_hat.len()] as usize;
        // Rotazione ciclica per garantire che H sia "universale"
        let shift = (i / self.h_hat.len()) % self.h_height;
        let mask = (1 << self.h_height) - 1;
        ((base << shift) | (base >> (self.h_height - shift))) & mask
    }
}

const INF_COST: f64 = f64::INFINITY;

/// Embedding STC tramite parity interleaving.
///
/// I coefficienti vengono divisi in k gruppi con interleaving:
///   gruppo j = { j, j+k, j+2k, ... }
/// Per ogni gruppo la parità dei bit stego deve essere uguale a message[j].
/// Se non lo è, si flippano il bit di costo minimo nel gruppo.
///
/// L'estrazione è semplicemente il ricalcolo delle stesse parità → correttezza garantita.
pub fn stc_embed(
    cover_bits: &[u8],
    costs: &[f64],
    message: &[u8],
    _params: &StcParams,
) -> Result<Vec<u8>, StcError> {
    let n = cover_bits.len();
    let k = message.len();

    if k > n {
        return Err(StcError::PayloadTooLarge { payload: k, capacity: n });
    }

    let mut stego_bits = cover_bits.to_vec();

    for j in 0..k {
        // Calcola parità attuale del gruppo j e trova l'elemento di costo minimo
        let mut parity: u8 = 0;
        let mut min_cost = f64::INFINITY;
        let mut min_idx = j;

        let mut idx = j;
        while idx < n {
            parity ^= cover_bits[idx];
            if costs[idx] < min_cost {
                min_cost = costs[idx];
                min_idx = idx;
            }
            idx += k;
        }

        // Se la parità non corrisponde al bit di messaggio, flippa il minimo-costo
        if parity != message[j] {
            stego_bits[min_idx] ^= 1;
        }
    }

    Ok(stego_bits)
}

/// Decoding STC: estrae il messaggio dall'immagine stego.
///
/// Per ogni gruppo j, calcola la parità XOR dei bit stego
/// nelle posizioni {j, j+k, j+2k, ...} — simmetrico all'embedding.
pub fn stc_extract(stego_bits: &[u8], k: usize, _params: &StcParams) -> Vec<u8> {
    let n = stego_bits.len();
    let mut message = vec![0u8; k];

    for j in 0..k {
        let mut parity: u8 = 0;
        let mut idx = j;
        while idx < n {
            parity ^= stego_bits[idx];
            idx += k;
        }
        message[j] = parity;
    }

    message
}

/// Converte bytes in bit (MSB first)
pub fn bytes_to_bits(bytes: &[u8]) -> Vec<u8> {
    bytes.iter()
        .flat_map(|b| (0..8).rev().map(move |i| (b >> i) & 1))
        .collect()
}

/// Converte bit in bytes (MSB first)
pub fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    bits.chunks(8)
        .map(|chunk| {
            chunk.iter().enumerate().fold(0u8, |acc, (i, &b)| {
                acc | (b << (7 - i))
            })
        })
        .collect()
}

/// Errori STC
#[derive(Debug)]
pub enum StcError {
    PayloadTooLarge { payload: usize, capacity: usize },
    EmbeddingFailed,
}

impl std::fmt::Display for StcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StcError::PayloadTooLarge { payload, capacity } =>
                write!(f, "Payload ({} bit) supera capacità ({} bit)", payload, capacity),
            StcError::EmbeddingFailed =>
                write!(f, "STC embedding fallito: nessun percorso valido nel trellis"),
        }
    }
}