// Optional encryption of a self-hosted model file: AES-256-CTR over the whole file, ciphertext as long as the
// plaintext. The IV is the initial 16-byte counter block, incremented as one 128-bit big-endian integer per 16-byte
// block (what `openssl enc -aes-256-ctr -K <key> -iv <iv>` does and WebCrypto's { name: 'AES-CTR', length: 128 }).
// There is no MAC: the SHA-256 of the plaintext, verified by model.js after decryption, is the integrity check.
// The key never goes into an error message or a log, and is not kept anywhere but in the decryptor's closure.

import { CzechSpeechError } from './events.js';

const KEY_HEX = /^[0-9a-fA-F]{64}$/;
const IV_HEX = /^[0-9a-fA-F]{32}$/;
const MASK_128 = (1n << 128n) - 1n;

/**
 * Checks the `encryption` of a model source and returns it normalised: { algorithm, key, iv } with lower-case hex.
 * TypeError (never quoting the key) for anything else than { algorithm: 'AES-256-CTR', key: 64 hex, iv: 32 hex }.
 */
export function normalizeEncryption(encryption) {
  if (!encryption || typeof encryption !== 'object') throw new TypeError("encryption must be { algorithm: 'AES-256-CTR', key, iv }");
  if (encryption.algorithm !== 'AES-256-CTR') throw new TypeError("encryption.algorithm must be 'AES-256-CTR'");
  if (typeof encryption.key !== 'string' || !KEY_HEX.test(encryption.key)) throw new TypeError('encryption.key must be 64 hex digits');
  if (typeof encryption.iv !== 'string' || !IV_HEX.test(encryption.iv)) throw new TypeError('encryption.iv must be 32 hex digits');
  return Object.freeze({ algorithm: 'AES-256-CTR', key: encryption.key.toLowerCase(), iv: encryption.iv.toLowerCase() });
}

const fromHex = (hex) => Uint8Array.from(hex.match(/../g), (b) => parseInt(b, 16));

/** The counter block for the 16-byte block `block` of a file: IV + block as one 128-bit big-endian integer (wrapping). */
export function counterBlock(ivHex, block) {
  let n = (BigInt(`0x${ivHex}`) + BigInt(block)) & MASK_128;
  const out = new Uint8Array(16);
  for (let i = 15; i >= 0; i--) {
    out[i] = Number(n & 0xffn);
    n >>= 8n;
  }
  return out;
}

/**
 * A function (bytes, offset) => Promise<Uint8Array> that decrypts `bytes`, which lie at byte `offset` of the file.
 * Stateless per call, so a download resumed at any offset (and chunks of any length) decrypts the same. `subtle` is
 * WebCrypto's SubtleCrypto (default: the global one).
 */
export function createDecryptor(encryption, subtle = globalThis.crypto && globalThis.crypto.subtle) {
  const enc = normalizeEncryption(encryption);
  if (!subtle) throw new CzechSpeechError('Encrypted models need WebCrypto (crypto.subtle), which is only available in a secure context (https)', { code: 'unsupported' });
  let keyPromise = null;
  const importKey = () => {
    keyPromise ||= (() => {
      const raw = fromHex(enc.key);
      return subtle.importKey('raw', raw, { name: 'AES-CTR' }, false, ['decrypt']).finally(() => raw.fill(0));
    })();
    return keyPromise;
  };
  return async (bytes, offset) => {
    if (bytes.length === 0) return new Uint8Array(0);
    const skip = offset % 16; // the keystream of a block that was started earlier is spent up to here
    let input = bytes;
    if (skip > 0) {
      input = new Uint8Array(skip + bytes.length);
      input.set(bytes, skip);
    }
    let plain;
    try {
      plain = new Uint8Array(await subtle.decrypt({ name: 'AES-CTR', counter: counterBlock(enc.iv, Math.floor(offset / 16)), length: 128 },
        await importKey(), input));
    } catch (err) {
      throw new CzechSpeechError(`Could not decrypt the model: ${err && err.name}`, { code: 'download' }); // no `cause`: nothing of the failure may carry the key
    }
    return skip > 0 ? plain.subarray(skip) : plain;
  };
}
