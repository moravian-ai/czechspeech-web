// Post-processing of the Emscripten glue of the browser modules (src/wasm/czechspeech-*.js). native/build-wasm.sh runs
// it on every module it builds; for modules built before, run it by hand: node scripts/fix-glue.mjs src/wasm/*.js
//
// Emscripten's ES-module output starts its pthread workers from its own file name:
//   new Worker(new URL("czechspeech-mt.js", import.meta.url), { type: "module", name: "em-pthread" })
// Bundlers emit the module under another name (Vite always adds a hash, webpack by default names assets by hash), so
// that URL points at nothing and the threads never start. The worker URL becomes the module's own URL,
//   new Worker(new URL(import.meta.url), { type: "module", name: "em-pthread" })
// which is the same file under whatever name it is served. It is a text change of the JS only; the .wasm is unchanged.
// The fallback `new URL("<name>.wasm", import.meta.url)` stays: it applies only without locateFile, and the SDK always
// passes one (src/engine-wasm.js).
//
// Idempotent: a fixed file is left as it is. A glue it cannot make independent of its name (another script resolved
// by name, pthread workers created in a form it does not know) is an error, so a build stops instead of shipping it.
//
// Usage: node scripts/fix-glue.mjs [--check] <glue.js>...
//   --check: change nothing; exit 1 when a file still starts its threads from a fixed name (or cannot be fixed)
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// new URL("<name>.js", import.meta.url): a script resolved next to the module by its file name
const SCRIPT_BY_NAME = /new URL\((["'])([^"']+\.m?js)\1\s*,\s*import\.meta\.url\)/g;
// the pthread workers started from the module's own URL
const OWN_URL_WORKER = /new Worker\(new URL\(import\.meta\.url\)\s*,\s*\{[^}]*name:\s*(["'])em-pthread\1/;

/** The scripts `text` resolves next to itself by a file name. */
export function fixedNameScripts(text) {
  return [...text.matchAll(SCRIPT_BY_NAME)].map((m) => m[2]);
}

/**
 * { text, changed }: `text` (the glue in `file`) with its pthread workers started from its own URL. Only the module
 * itself is rewritten: its own file name in its own directory ("name.js" or "./name.js"); the same name elsewhere
 * ("sub/name.js") is another file, an error like any other script loaded by name.
 */
export function fixGlue(text, file) {
  const own = path.basename(file);
  const isOwn = (name) => name === own || name === `./${own}`;
  const others = fixedNameScripts(text).filter((name) => !isOwn(name));
  if (others.length > 0) throw new Error(`${own}: loads other scripts by name: ${others.join(', ')}`);
  const pthreads = /(["'])em-pthread\1/.test(text);
  if (/-mt\.js$/.test(own) && !pthreads) {
    throw new Error(`${own}: the multi-threaded glue creates no pthread workers (em-pthread)`);
  }
  const fixed = text.replace(SCRIPT_BY_NAME, 'new URL(import.meta.url)');
  if (pthreads && !OWN_URL_WORKER.test(fixed)) {
    throw new Error(`${own}: its pthread workers are not started from new URL(import.meta.url)`);
  }
  return { text: fixed, changed: fixed !== text };
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const check = process.argv.includes('--check');
  const files = process.argv.slice(2).filter((a) => a !== '--check');
  let failures = 0;
  for (const file of files) {
    try {
      const { text, changed } = fixGlue(fs.readFileSync(file, 'utf8'), file);
      if (check && changed) {
        console.error(`${path.basename(file)}: starts its pthread workers from a fixed file name; run node scripts/fix-glue.mjs`);
        failures++;
      } else if (changed) {
        fs.writeFileSync(file, text);
        console.log(`${path.basename(file)}: pthread workers start from the module's own URL`);
      } else {
        console.log(`${path.basename(file)}: nothing to change`);
      }
    } catch (err) {
      console.error(err.message);
      failures++;
    }
  }
  process.exit(failures ? 1 : 0);
}
