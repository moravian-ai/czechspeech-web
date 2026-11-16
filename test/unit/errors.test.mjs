// Error codes: every CzechSpeechError the package produces has one of the codes documented in the d.ts
// (CzechSpeechErrorCode), whatever the code of the error behind it (a DOMException's numeric code, Node's 'ENOENT').
// The paths: the source of every `code:` in src/, the error class itself, and the worker failures with the fake engine.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { CzechSpeech, CzechSpeechModel, CzechSpeechError } from '../../src/index.js';
import { ArrayAudioSource, ROOT, Signals, TestEnv, nodeWorkerFactory } from '../support/env.mjs';

const dts = fs.readFileSync(path.join(ROOT, 'types/index.d.ts'), 'utf8');
const DOCUMENTED = [.../export type CzechSpeechErrorCode =([^;]+);/.exec(dts)[1].matchAll(/'([a-z-]+)'/g)].map((m) => m[1]);
const documented = (code) => DOCUMENTED.includes(code);

const fakeEntry = new URL('../support/fake-worker-entry.mjs', import.meta.url);
const vadModel = fs.readFileSync(TestEnv.vadAsset);
const utterance = Signals.concat(Signals.zeros(0.5), Signals.tone(1.0), Signals.zeros(1.0));

test('the documented codes are the ones the package knows', async () => {
  assert.equal(DOCUMENTED.length, 12);
  const { ERROR_CODES } = await import('../../src/events.js');
  assert.deepEqual(ERROR_CODES, DOCUMENTED);
  assert.ok(Object.isFrozen(ERROR_CODES));
});

test('CzechSpeechError keeps a documented code and replaces any other with runtime', () => {
  for (const code of DOCUMENTED) assert.equal(new CzechSpeechError('x', { code }).code, code);
  for (const code of [undefined, null, 25, 'ENOENT', '', 'Runtime', {}]) {
    assert.equal(new CzechSpeechError('x', { code }).code, 'runtime', String(code));
  }
  assert.equal(new CzechSpeechError('x').code, 'runtime');
});

test('every code in src/ is documented, and every code that comes from elsewhere is normalised', () => {
  const dir = path.join(ROOT, 'src');
  const sites = [];
  for (const file of fs.readdirSync(dir).filter((f) => f.endsWith('.js') && f !== 'worklet.js')) {
    fs.readFileSync(path.join(dir, file), 'utf8').split('\n').forEach((line, i) => {
      if (/^\s*(\/\/|\*)/.test(line)) return; // comments
      for (const m of line.matchAll(/\bcode:\s*([^,}]+)/g)) sites.push({ where: `src/${file}:${i + 1}`, value: m[1].trim(), line });
    });
  }
  assert.ok(sites.length > 50, `${sites.length} sites`);
  for (const { where, value, line } of sites) {
    const literal = /^'([^']*)'$/.exec(value);
    if (literal) {
      assert.ok(documented(literal[1]), `${where}: '${literal[1]}' is not a documented code`);
    } else {
      // a code from elsewhere: through errorCode() in the workers, or the CzechSpeechError constructor on the page
      assert.ok(/^errorCode\(/.test(value) || /new CzechSpeechError\(/.test(line), `${where}: code ${value} is not normalised`);
    }
  }
});

async function setup(engine) {
  const factory = nodeWorkerFactory(fakeEntry);
  const speech = await CzechSpeech.create({ model: CzechSpeechModel._fromPath('fake-model.bin', 1000), log: false,
    _createWorker: factory, _vadModel: vadModel, _engineOptions: engine });
  return { speech, factory };
}

const collect = async (iterable) => {
  const events = [];
  for await (const ev of iterable) events.push(ev);
  return events;
};

const failure = (p) => p.then((v) => assert.fail(`resolved ${v}`), (e) => e);

test('worker failures reach the app with documented codes, whatever code the error behind them had', async () => {
  const seen = [];
  const check = (what, e, expected) => {
    seen.push(`${what}: ${e && e.code}`);
    assert.ok(e instanceof CzechSpeechError, `${what}: ${e}`);
    assert.equal(e.code, expected, `${what}: code ${JSON.stringify(e.code)}`);
  };
  for (const errorKind of ['dom', 'node', undefined]) {
    // the engine does not load: the recogniser's start-up fails (transcribe), both workers' (listen)
    {
      const { speech } = await setup({ failLoad: true, errorKind });
      check(`${errorKind}: transcribe, engine load`, await failure(speech.transcribe(Signals.tone(1.0))), 'runtime');
      const source = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
      check(`${errorKind}: listen, engine load`, await failure(collect(speech.listen({ _source: source }))), 'runtime');
      await speech.close();
    }
    // the recognition throws: transcribe() rejects, listen() gives an error event and goes on
    {
      const { speech } = await setup({ failIn: 'transcribe', errorKind });
      check(`${errorKind}: transcribe, recognition`, await failure(speech.transcribe(Signals.tone(1.0))), 'recognition');
      const events = await collect(speech.listen({ _source: new ArrayAudioSource(utterance) }));
      const errors = events.filter((e) => e.type === 'error');
      assert.equal(errors.length, 1);
      check(`${errorKind}: listen, recognition`, errors[0].error, 'recognition');
      await speech.close();
    }
    // the engine reports a successful recognition without a language: an inconsistency of the native layer, so a
    // failed recognition (no language is made up), not a transcript with a guessed one
    {
      const { speech } = await setup({ noLanguage: true, errorKind });
      check(`${errorKind}: transcribe, empty language`, await failure(speech.transcribe(Signals.tone(1.0))), 'recognition');
      const events = await collect(speech.listen({ _source: new ArrayAudioSource(utterance) }));
      assert.equal(events.filter((e) => e.type === 'transcript').length, 0);
      const errors = events.filter((e) => e.type === 'error');
      assert.equal(errors.length, 1);
      check(`${errorKind}: listen, empty language`, errors[0].error, 'recognition');
      await speech.close();
    }
    // the worker dies of an uncaught error (its onerror)
    {
      const { speech, factory } = await setup({ failIn: 'crash', errorKind });
      check(`${errorKind}: transcribe, worker crash`, await failure(speech.transcribe(Signals.tone(1.0))), 'runtime');
      assert.ok(factory.created.every((w) => w.terminated));
      await speech.close();
    }
  }
  assert.ok(seen.every((s) => documented(s.split(': ').pop())), seen.join('\n'));
});

// ---- no raw errors: what the platform throws reaches the app as a CzechSpeechError, the original as its cause ----

test('create(): a VAD asset that cannot be fetched or read is a runtime error, not a raw TypeError', async () => {
  const model = CzechSpeechModel._fromPath('fake-model.bin', 1000);
  const noWorkers = () => assert.fail('no worker is started by create()'); // Node.js has no Web Workers
  const saved = globalThis.fetch;
  const offline = new TypeError('fetch failed');
  const broken = new TypeError('terminated');
  try {
    globalThis.fetch = async () => { throw offline; };
    const a = await failure(CzechSpeech.create({ model, log: false, _createWorker: noWorkers }));
    assert.ok(a instanceof CzechSpeechError && a.code === 'runtime' && a.cause === offline, `${a}`);
    assert.match(a.message, /Silero VAD/);
    globalThis.fetch = async () => ({ ok: true, status: 200, arrayBuffer: async () => { throw broken; } });
    const b = await failure(CzechSpeech.create({ model, log: false, _createWorker: noWorkers }));
    assert.ok(b instanceof CzechSpeechError && b.code === 'runtime' && b.cause === broken, `${b}`);
    globalThis.fetch = async () => new Response('', { status: 404 });
    const c = await failure(CzechSpeech.create({ model, log: false, _createWorker: noWorkers }));
    assert.ok(c instanceof CzechSpeechError && c.code === 'runtime' && /HTTP 404/.test(c.message), `${c}`);
  } finally {
    globalThis.fetch = saved;
  }
  // invalid arguments stay TypeError and RangeError (programming errors)
  await assert.rejects(CzechSpeech.create({}), TypeError);
  await assert.rejects(CzechSpeech.create({ model, config: { threshold: 2 }, _vadModel: vadModel }), RangeError);
});

/** nodeWorkerFactory(fakeEntry) whose workers of `role` throw `error` from postMessage for messages of `type`. */
function refusingWorkers(role, type, error) {
  const inner = nodeWorkerFactory(fakeEntry);
  const factory = (r) => {
    const w = inner(r);
    if (r === role) {
      const post = w.postMessage;
      w.postMessage = (m, t) => {
        if (m.type === type) throw error;
        post.call(w, m, t);
      };
    }
    return w;
  };
  factory.created = inner.created;
  return factory;
}

test('a message the browser refuses to send is a runtime error with the browser\'s error as cause', async () => {
  const error = new DOMException('could not be cloned', 'DataCloneError');
  // transcribe(): the samples to the recogniser
  {
    const factory = refusingWorkers('recognizer', 'transcribe', error);
    const speech = await CzechSpeech.create({ model: CzechSpeechModel._fromPath('fake-model.bin', 1000), log: false,
      _createWorker: factory, _vadModel: vadModel });
    const e = await failure(speech.transcribe(Signals.tone(1.0)));
    assert.ok(e instanceof CzechSpeechError && e.code === 'runtime' && e.cause === error, `${e}`);
    assert.ok(factory.created.every((w) => w.terminated));
    await speech.close();
  }
  // listen(): the audio port to the capture worker
  {
    const factory = refusingWorkers('capture', 'start', error);
    const speech = await CzechSpeech.create({ model: CzechSpeechModel._fromPath('fake-model.bin', 1000), log: false,
      _createWorker: factory, _vadModel: vadModel });
    const source = new ArrayAudioSource(utterance, { tail: () => 0, frameDelayMs: 1 });
    const e = await failure(collect(speech.listen({ _source: source })));
    assert.ok(e instanceof CzechSpeechError && e.code === 'runtime' && e.cause === error, `${e}`);
    assert.ok(source.closed);
    assert.ok(factory.created.every((w) => w.terminated));
    await speech.close();
  }
});
