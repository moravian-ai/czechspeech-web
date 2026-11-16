// Static guards on the repository itself: file modes in the git index and package.json fields that matter to the
// people who install the package or work on it.
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from '../support/env.mjs';

const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));

function gitIndex(pattern) {
  try {
    return execFileSync('git', ['ls-files', '-s', '--', pattern], { cwd: ROOT, encoding: 'utf8' });
  } catch {
    return null;
  }
}

describe('static guards: the repository and package.json', () => {
  test('the shell scripts are executable in the git index (100755)', (t) => {
    const out = gitIndex('native/*.sh');
    if (out === null) {
      t.skip('not a git checkout');
      return;
    }
    const entries = out.trim().split('\n').map((l) => /^(\d{6}) [0-9a-f]+ \d\t(.+)$/.exec(l)).map((m) => ({ mode: m[1], file: m[2] }));
    assert.deepEqual(entries.map((e) => e.file).sort(), ['native/build-wasm.sh', 'native/strip-pthread.sh']);
    for (const e of entries) assert.equal(e.mode, '100755', `${e.file} is ${e.mode} in the index`);
  });

  test('no engines (the package runs in browsers); development needs Node.js 22, in devEngines', () => {
    // engines would make engine-strict npm installs and Yarn refuse the package on an older Node.js, although nothing
    // of it runs in Node.js; devEngines applies to this repository only
    assert.equal(pkg.engines, undefined);
    assert.deepEqual(pkg.devEngines, { runtime: { name: 'node', version: '>=22', onFail: 'error' } });
  });
});
