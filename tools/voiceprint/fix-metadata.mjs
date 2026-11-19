// Rewrites the descriptive metadata (ModelProto field 14, metadata_props) of the voiceprint ONNX file. Every other
// top-level field, the graph and so the weights, is copied byte for byte in the original order.
import fs from 'node:fs';
import crypto from 'node:crypto';
import { pathToFileURL } from 'node:url';

export const VOICEPRINT_METADATA = Object.freeze([
  ['framework', '3d-speaker'],
  ['language', 'Czech'],
  ['url', 'https://moravian.ai'],
  ['comment', 'CzechSpeech Voiceprint (voiceprint-cs) by Moravian AI: 3D-Speaker ERes2Net (iic/speech_eres2net_sv_en_voxceleb_16k v1.0.2, Apache-2.0) fine-tuned on ParCzech4Speech 1.0 (UFAL MFF UK, CC BY); weights mixed 75 % fine-tuned and 25 % original. Modified from the original model.'],
  ['sample_rate', '16000'],
  ['output_dim', '192'],
  ['normalize_samples', '1'],
  ['feature_normalize_type', 'global-mean'],
]);

const INPUT_SHA256 = '34ecdabf67fabfe66924e1252ad45a3585be874849740ac7d0ac0d29a8cdeecc';

function readVarint(bytes, pos) {
  let result = 0;
  let mul = 1;
  for (let i = 0; i < 10; i++) {
    if (pos >= bytes.length) throw new RangeError('truncated varint');
    const b = bytes[pos++];
    result += (b & 0x7f) * mul;
    if (!(b & 0x80)) return [result, pos];
    mul *= 128;
  }
  throw new RangeError('varint too long');
}

/** Top-level fields of a ModelProto: { field, wire, start, end } where [start, end) covers tag and payload. */
export function readModelProto(bytes) {
  const fields = [];
  let pos = 0;
  while (pos < bytes.length) {
    const start = pos;
    let tag;
    [tag, pos] = readVarint(bytes, pos);
    const wire = tag & 7;
    if (wire === 0) {
      [, pos] = readVarint(bytes, pos);
    } else if (wire === 2) {
      let len;
      [len, pos] = readVarint(bytes, pos);
      pos += len;
      if (pos > bytes.length) throw new RangeError('truncated field');
    } else {
      throw new RangeError(`unsupported wire type ${wire} at the top level`);
    }
    fields.push({ field: Math.floor(tag / 8), wire, start, end: pos });
  }
  return fields;
}

function varintBytes(n) {
  const out = [];
  do { let b = n % 128; n = Math.floor(n / 128); if (n) b |= 0x80; out.push(b); } while (n);
  return out;
}

const encoder = new TextEncoder();
function lengthDelimited(field, payload) {
  return [...varintBytes(field * 8 + 2), ...varintBytes(payload.length), ...payload];
}

/** Drops every field 14 and appends the given entries as new field 14 (StringStringEntryProto: 1 key, 2 value). */
export function rewriteMetadata(bytes, entries) {
  const parts = [];
  for (const f of readModelProto(bytes)) if (f.field !== 14) parts.push(bytes.subarray(f.start, f.end));
  for (const [k, v] of entries) {
    const inner = [...lengthDelimited(1, encoder.encode(k)), ...lengthDelimited(2, encoder.encode(v))];
    parts.push(Uint8Array.from(lengthDelimited(14, inner)));
  }
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [input, output] = process.argv.slice(2);
  if (!input || !output) { console.error('usage: node fix-metadata.mjs <input.onnx> <output.onnx>'); process.exit(2); }
  const bytes = fs.readFileSync(input);
  const sha = crypto.createHash('sha256').update(bytes).digest('hex');
  if (sha !== INPUT_SHA256) { console.error(`input sha256 ${sha} is not the expected ${INPUT_SHA256}`); process.exit(1); }
  const out = rewriteMetadata(bytes, VOICEPRINT_METADATA);
  fs.writeFileSync(output, out);
  console.log(`size=${out.length} sha256=${crypto.createHash('sha256').update(out).digest('hex')}`);
}
