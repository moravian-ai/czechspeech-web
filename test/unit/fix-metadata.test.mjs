import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readModelProto, rewriteMetadata, VOICEPRINT_METADATA } from '../../tools/voiceprint/fix-metadata.mjs';

// Minimal protobuf writer for the test: varint, length-delimited field.
const varint = (n) => { const out = []; do { let b = n & 0x7f; n >>>= 7; if (n) b |= 0x80; out.push(b); } while (n); return out; };
const ld = (field, bytes) => [...varint((field << 3) | 2), ...varint(bytes.length), ...bytes];
const str = (s) => [...new TextEncoder().encode(s)];
const entry = (k, v) => ld(14, [...ld(1, str(k)), ...ld(2, str(v))]);
const model = Uint8Array.from([
  ...varint((1 << 3) | 0), 7,                    // ir_version = 7
  ...ld(2, str('pytorch')),                      // producer_name
  ...ld(7, [1, 2, 3, 4, 5]),                     // graph (opaque bytes)
  ...entry('language', 'English'),
  ...entry('comment', 'This model is from iic/x'),
  ...ld(8, [9, 9]),                              // opset_import
]);

test('rewriteMetadata keeps every other top-level field byte for byte and in order', () => {
  const out = rewriteMetadata(model, [['language', 'Czech']]);
  const before = readModelProto(model).filter((f) => f.field !== 14).map((f) => [...model.subarray(f.start, f.end)]);
  const after = readModelProto(out).filter((f) => f.field !== 14).map((f) => [...out.subarray(f.start, f.end)]);
  assert.deepEqual(after, before);
});

test('rewriteMetadata replaces all metadata entries with the given ones', () => {
  const out = rewriteMetadata(model, [['language', 'Czech'], ['url', 'https://moravian.ai']]);
  const meta = readModelProto(out).filter((f) => f.field === 14);
  assert.equal(meta.length, 2);
  const text = new TextDecoder().decode(out);
  assert.ok(!text.includes('English') && !text.includes('iic/x'));
  assert.ok(text.includes('Czech') && text.includes('https://moravian.ai'));
});

test('VOICEPRINT_METADATA keeps the keys sherpa-onnx reads and names the modification', () => {
  const m = Object.fromEntries(VOICEPRINT_METADATA);
  assert.deepEqual([m.framework, m.sample_rate, m.output_dim, m.normalize_samples, m.feature_normalize_type],
    ['3d-speaker', '16000', '192', '1', 'global-mean']);
  assert.match(m.comment, /ParCzech4Speech/);
  assert.match(m.comment, /Modified from the original model/);
});
