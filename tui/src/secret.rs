//! Hidden-content gate. Source never holds plaintext — only digests.

use sha2::{Digest, Sha256};
use std::path::Path;

const KEY_SALT: &[u8] = b"\x00sub-zero-easter-v1";

pub fn digest_of(phrase: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(phrase.as_bytes());
    h.finalize().into()
}

pub fn digests_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

pub fn verify(phrase: &str, expected: &[u8; 32]) -> bool {
    digests_eq(&digest_of(phrase), expected)
}

pub fn derive_key(phrase: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(phrase.as_bytes());
    h.update(KEY_SALT);
    h.finalize().into()
}

pub fn xor_stream(data: &mut [u8], key: &[u8; 32]) {
    let mut counter: u64 = 0;
    let mut offset = 0;
    while offset < data.len() {
        let mut h = Sha256::new();
        h.update(key);
        h.update(counter.to_le_bytes());
        let block: [u8; 32] = h.finalize().into();
        let take = (data.len() - offset).min(32);
        for i in 0..take {
            data[offset + i] ^= block[i];
        }
        offset += take;
        counter = counter.wrapping_add(1);
    }
}

pub fn decrypt_file(path: &Path, phrase: &str) -> std::io::Result<Vec<u8>> {
    let mut data = std::fs::read(path)?;
    let key = derive_key(phrase);
    xor_stream(&mut data, &key);
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_is_deterministic() {
        assert_eq!(digest_of("hello"), digest_of("hello"));
        assert_ne!(digest_of("hello"), digest_of("Hello"));
    }

    #[test]
    fn xor_stream_round_trip() {
        let key = derive_key("test phrase");
        let original = b"The quick brown fox jumps over the lazy dog.".to_vec();
        let mut encrypted = original.clone();
        xor_stream(&mut encrypted, &key);
        assert_ne!(encrypted, original);
        let mut decrypted = encrypted.clone();
        xor_stream(&mut decrypted, &key);
        assert_eq!(decrypted, original);
    }

    #[test]
    fn wrong_phrase_does_not_decrypt() {
        let mut data = b"secret manifest contents".to_vec();
        xor_stream(&mut data, &derive_key("correct"));
        let copy = data.clone();
        let mut wrong_try = data;
        xor_stream(&mut wrong_try, &derive_key("incorrect"));
        assert_ne!(&wrong_try, b"secret manifest contents");
        assert_ne!(wrong_try, copy);
    }

    #[test]
    fn verify_accepts_match_rejects_others() {
        let d = digest_of("opensesame");
        assert!(verify("opensesame", &d));
        assert!(!verify("OpenSesame", &d));
        assert!(!verify("", &d));
    }

    #[test]
    fn digests_eq_is_constant_time_correct() {
        let a = digest_of("a");
        let b = digest_of("a");
        let c = digest_of("b");
        assert!(digests_eq(&a, &b));
        assert!(!digests_eq(&a, &c));
    }

    #[test]
    fn long_buffer_decrypts_correctly() {
        let key = derive_key("counter rollover test");
        let original: Vec<u8> = (0..1024u32).map(|i| (i & 0xff) as u8).collect();
        let mut buf = original.clone();
        xor_stream(&mut buf, &key);
        xor_stream(&mut buf, &key);
        assert_eq!(buf, original);
    }
}
