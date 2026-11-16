// Guards the public API: what the package exports at run time (every export, and the public members of every
// exported class) must equal the listing in types/public-api.txt, and every export must be declared in
// types/index.d.ts. A change of the API is then a change of that file in the same commit, where a reviewer sees it.
// To accept a change: CZECHSPEECH_UPDATE_API=1 node --test test/unit/api.test.mjs, and commit the file.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from '../support/env.mjs';
import * as api from '../../src/index.js';

const FILE = path.join(ROOT, 'types/public-api.txt');

function describeValue(value) {
  if (typeof value === 'function' && /^class\s/.test(Function.prototype.toString.call(value))) return 'class';
  if (typeof value === 'function') return 'function';
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  if (typeof value === 'object') return `object { ${Object.keys(value).sort().join(', ')} }`;
  return `${typeof value} ${JSON.stringify(value)}`;
}

function members(target, skip) {
  return Object.getOwnPropertyNames(target).filter((n) => !skip.includes(n)).sort().map((name) => {
    const d = Object.getOwnPropertyDescriptor(target, name);
    if (typeof d.value === 'function') return `${name}()`;
    if (d.get || d.set) return `${name} { ${d.get ? 'get' : ''}${d.get && d.set ? ' ' : ''}${d.set ? 'set' : ''} }`;
    return name;
  });
}

function listing() {
  const lines = [];
  for (const name of Object.keys(api).sort()) {
    const value = api[name];
    const kind = describeValue(value);
    // VERSION changes with every release; it is not part of the API's shape
    lines.push(name === 'VERSION' ? 'VERSION: string' : `${name}: ${kind}`);
    if (kind !== 'class') continue;
    for (const m of members(value, ['length', 'name', 'prototype'])) lines.push(`    static ${m}`);
    for (const m of members(value.prototype, ['constructor'])) lines.push(`    ${m}`);
  }
  return `${lines.join('\n')}\n`;
}

test('the exports and the public members of the exported classes are the recorded ones', () => {
  const actual = listing();
  if (process.env.CZECHSPEECH_UPDATE_API === '1') fs.writeFileSync(FILE, actual);
  assert.ok(fs.existsSync(FILE), `${FILE} is missing; run this test once with CZECHSPEECH_UPDATE_API=1`);
  assert.equal(actual, fs.readFileSync(FILE, 'utf8').replace(/\r\n/g, '\n'),
    'the public API changed; if that is intended, run this test with CZECHSPEECH_UPDATE_API=1 and commit types/public-api.txt');
});

test('every export is declared in types/index.d.ts', () => {
  const dts = fs.readFileSync(path.join(ROOT, 'types/index.d.ts'), 'utf8');
  for (const name of Object.keys(api)) {
    assert.match(dts, new RegExp(`export declare (const|class|function) ${name}\\b`), `${name} is not declared in types/index.d.ts`);
  }
});
