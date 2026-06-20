//! Authenticated hidden-content envelopes.
//!
//! The shipped binary does not contain unlock digests or plaintext assets.
//! A phrase is used only to derive a slot key for the encrypted manifest;
//! asset blobs then decrypt with that key.

use argon2::{Algorithm, Argon2, Params, Version};
use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Key, Nonce,
};
use sha2::{Digest, Sha256};
use std::io::{Error, ErrorKind};
use std::path::Path;
use zeroize::Zeroize;

const MANIFEST_MAGIC: &[u8; 8] = b"SZEE2M\0\0";
const ASSET_MAGIC: &[u8; 8] = b"SZEE2A\0\0";
const LEGACY_KEY_SALT: &[u8] = b"\x00voidex-easter-v1";
const SALT_LEN: usize = 16;
const NONCE_LEN: usize = 12;
const KEY_LEN: usize = 32;

#[cfg(not(test))]
const ARGON2_MEMORY_KIB: u32 = 64 * 1024;
#[cfg(test)]
const ARGON2_MEMORY_KIB: u32 = 64;
const ARGON2_PASSES: u32 = 3;
const ARGON2_LANES: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SecretFormat {
    V2,
    LegacyXor,
}

pub struct SecretKey {
    bytes: [u8; KEY_LEN],
    format: SecretFormat,
}

impl SecretKey {
    fn cipher(&self) -> ChaCha20Poly1305 {
        ChaCha20Poly1305::new(Key::from_slice(&self.bytes))
    }
}

impl std::fmt::Debug for SecretKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecretKey")
            .field("format", &self.format)
            .field("bytes", &"<redacted>")
            .finish()
    }
}

impl Drop for SecretKey {
    fn drop(&mut self) {
        self.bytes.zeroize();
    }
}

pub fn decrypt_manifest_file(path: &Path, phrase: &str) -> std::io::Result<(Vec<u8>, SecretKey)> {
    let data = std::fs::read(path)?;
    decrypt_manifest(&data, phrase)
}

pub fn decrypt_asset_file(path: &Path, key: &SecretKey) -> std::io::Result<Vec<u8>> {
    let data = std::fs::read(path)?;
    decrypt_asset(&data, key)
}

fn decrypt_manifest(data: &[u8], phrase: &str) -> std::io::Result<(Vec<u8>, SecretKey)> {
    if !data.starts_with(MANIFEST_MAGIC) {
        let key = derive_legacy_key(phrase);
        let plaintext = legacy_xor(data, &key.bytes);
        return Ok((plaintext, key));
    }

    let min_len = MANIFEST_MAGIC.len() + SALT_LEN + NONCE_LEN;
    if data.len() <= min_len {
        return Err(invalid_data("invalid hidden manifest envelope"));
    }

    let salt_start = MANIFEST_MAGIC.len();
    let nonce_start = salt_start + SALT_LEN;
    let ciphertext_start = nonce_start + NONCE_LEN;
    let salt = &data[salt_start..nonce_start];
    let nonce = &data[nonce_start..ciphertext_start];
    let ciphertext = &data[ciphertext_start..];

    let key = derive_key(phrase, salt)?;
    let plaintext = decrypt_with_key(&key, nonce, ciphertext)?;
    Ok((plaintext, key))
}

fn decrypt_asset(data: &[u8], key: &SecretKey) -> std::io::Result<Vec<u8>> {
    if key.format == SecretFormat::LegacyXor {
        return Ok(legacy_xor(data, &key.bytes));
    }

    let min_len = ASSET_MAGIC.len() + NONCE_LEN;
    if data.len() <= min_len || &data[..ASSET_MAGIC.len()] != ASSET_MAGIC {
        return Err(invalid_data("invalid hidden asset envelope"));
    }

    let nonce_start = ASSET_MAGIC.len();
    let ciphertext_start = nonce_start + NONCE_LEN;
    let nonce = &data[nonce_start..ciphertext_start];
    let ciphertext = &data[ciphertext_start..];
    decrypt_with_key(key, nonce, ciphertext)
}

fn derive_key(phrase: &str, salt: &[u8]) -> std::io::Result<SecretKey> {
    let params = Params::new(
        ARGON2_MEMORY_KIB,
        ARGON2_PASSES,
        ARGON2_LANES,
        Some(KEY_LEN),
    )
    .map_err(|error| invalid_data(format!("invalid Argon2 parameters: {error}")))?;
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

    let mut key = [0u8; KEY_LEN];
    if let Err(error) = argon2.hash_password_into(phrase.as_bytes(), salt, &mut key) {
        key.zeroize();
        return Err(invalid_data(format!(
            "Argon2 key derivation failed: {error}"
        )));
    }
    Ok(SecretKey {
        bytes: key,
        format: SecretFormat::V2,
    })
}

fn decrypt_with_key(key: &SecretKey, nonce: &[u8], ciphertext: &[u8]) -> std::io::Result<Vec<u8>> {
    key.cipher()
        .decrypt(Nonce::from_slice(nonce), ciphertext)
        .map_err(|_| invalid_data("hidden asset authentication failed"))
}

fn derive_legacy_key(phrase: &str) -> SecretKey {
    let mut h = Sha256::new();
    h.update(phrase.as_bytes());
    h.update(LEGACY_KEY_SALT);
    SecretKey {
        bytes: h.finalize().into(),
        format: SecretFormat::LegacyXor,
    }
}

fn legacy_xor(data: &[u8], key: &[u8; KEY_LEN]) -> Vec<u8> {
    let mut out = data.to_vec();
    let mut counter = 0u64;
    let mut offset = 0usize;

    while offset < out.len() {
        let mut h = Sha256::new();
        h.update(key);
        h.update(counter.to_le_bytes());
        let block: [u8; KEY_LEN] = h.finalize().into();
        let take = (out.len() - offset).min(KEY_LEN);
        for i in 0..take {
            out[offset + i] ^= block[i];
        }
        offset += take;
        counter = counter.wrapping_add(1);
    }

    out
}

fn invalid_data(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encrypt_manifest_for_test(phrase: &str, plaintext: &[u8]) -> Vec<u8> {
        let salt = [7u8; SALT_LEN];
        let nonce = [9u8; NONCE_LEN];
        let key = derive_key(phrase, &salt).expect("test key should derive");
        let ciphertext = key
            .cipher()
            .encrypt(Nonce::from_slice(&nonce), plaintext)
            .expect("test encrypt should succeed");

        let mut out = Vec::new();
        out.extend_from_slice(MANIFEST_MAGIC);
        out.extend_from_slice(&salt);
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&ciphertext);
        out
    }

    fn encrypt_asset_for_test(key: &SecretKey, plaintext: &[u8]) -> Vec<u8> {
        let nonce = [11u8; NONCE_LEN];
        let ciphertext = key
            .cipher()
            .encrypt(Nonce::from_slice(&nonce), plaintext)
            .expect("test encrypt should succeed");

        let mut out = Vec::new();
        out.extend_from_slice(ASSET_MAGIC);
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&ciphertext);
        out
    }

    fn encrypt_legacy_for_test(phrase: &str, plaintext: &[u8]) -> Vec<u8> {
        let key = derive_legacy_key(phrase);
        legacy_xor(plaintext, &key.bytes)
    }

    #[test]
    fn manifest_unlock_derives_asset_key() {
        let envelope = encrypt_manifest_for_test("correct horse", b"{\"items\":[]}");
        let (manifest, key) = decrypt_manifest(&envelope, "correct horse").expect("unlock");
        assert_eq!(manifest, b"{\"items\":[]}");

        let asset = encrypt_asset_for_test(&key, b"secret asset");
        let plaintext = decrypt_asset(&asset, &key).expect("asset decrypt");
        assert_eq!(plaintext, b"secret asset");
    }

    #[test]
    fn wrong_phrase_rejects_manifest() {
        let envelope = encrypt_manifest_for_test("correct horse", b"secret manifest");
        assert!(decrypt_manifest(&envelope, "wrong horse").is_err());
    }

    #[test]
    fn tampered_asset_rejects_authentication() {
        let envelope = encrypt_manifest_for_test("correct horse", b"manifest");
        let (_, key) = decrypt_manifest(&envelope, "correct horse").expect("unlock");
        let mut asset = encrypt_asset_for_test(&key, b"secret asset");
        let last = asset.last_mut().expect("ciphertext byte");
        *last ^= 0x55;
        assert!(decrypt_asset(&asset, &key).is_err());
    }

    #[test]
    fn legacy_xor_manifest_and_asset_still_unlock() {
        let envelope = encrypt_legacy_for_test("old phrase", b"{\"items\":[]}");
        let (manifest, key) = decrypt_manifest(&envelope, "old phrase").expect("legacy unlock");
        assert_eq!(manifest, b"{\"items\":[]}");

        let asset = encrypt_legacy_for_test("old phrase", b"legacy asset");
        let plaintext = decrypt_asset(&asset, &key).expect("legacy asset decrypt");
        assert_eq!(plaintext, b"legacy asset");
    }
}
