#!/usr/bin/env node
// Tiny static dev server for the example and the browser tests (Node.js stdlib only).
//
// - Serves the repository root on 127.0.0.1 (open http://localhost:8790/examples/basic/).
// - Sends the headers that make a page cross-origin isolated (multi-threaded WebAssembly):
//     Cross-Origin-Opener-Policy: same-origin
//     Cross-Origin-Embedder-Policy: require-corp
//   plus Cross-Origin-Resource-Policy: same-origin on every file. A document requested with ?coi=0 gets neither
//   COOP nor COEP, so the same origin (and the same cached model) can also show the single-threaded fallback.
// - Correct MIME types (application/wasm for .wasm), HTTP Range requests (resumable model download), no caching.
// - Optional mounts: --model <file> at /models/<file name> (so pages need not download 474 MB from Hugging Face),
//   --fixtures <dir> at /fixtures/ (test audio for test/e2e); startServer({ mounts: [{ prefix, dir }] }) for more
//   (the ParCzech audio of test/e2e/accuracy.mjs).
//
// Usage: node scripts/serve.mjs [--port 8790] [--model <file>] [--fixtures <dir>] [--no-isolation]
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.wasm': 'application/wasm',
  '.bin': 'application/octet-stream', '.wav': 'audio/wav', '.svg': 'image/svg+xml', '.md': 'text/markdown; charset=utf-8',
  '.map': 'application/json', '.txt': 'text/plain; charset=utf-8',
};

export function startServer({ port = 8790, host = '127.0.0.1', model = null, fixtures = null, mounts: extra = [], isolation = true,
  quiet = false } = {}) {
  const mounts = [];
  if (model) mounts.push({ prefix: `/models/${path.basename(model)}`, file: path.resolve(model) });
  if (fixtures) mounts.push({ prefix: '/fixtures/', dir: path.resolve(fixtures) });
  for (const m of extra) mounts.push({ prefix: m.prefix, dir: path.resolve(m.dir) });
  const stats = { requests: 0, bytes: 0 };

  const resolve = (pathname) => {
    for (const m of mounts) {
      if (m.file && pathname === m.prefix) return m.file;
      if (m.dir && pathname.startsWith(m.prefix)) return inside(m.dir, pathname.slice(m.prefix.length));
    }
    return inside(ROOT, pathname.slice(1));
  };

  const server = http.createServer((req, res) => {
    stats.requests++;
    const url = new URL(req.url, 'http://localhost');
    const file = resolve(decodeURIComponent(url.pathname));
    const isDocument = /\.html?$/.test(url.pathname) || url.pathname.endsWith('/');
    const headers = { 'Cache-Control': 'no-store', 'Cross-Origin-Resource-Policy': 'same-origin',
      'X-Content-Type-Options': 'nosniff' };
    if (isolation && !(isDocument && url.searchParams.get('coi') === '0')) {
      headers['Cross-Origin-Opener-Policy'] = 'same-origin';
      headers['Cross-Origin-Embedder-Policy'] = 'require-corp';
    }
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(405, headers);
      res.end();
      return;
    }
    if (url.pathname === '/favicon.ico') {
      // Chromium's own favicon request (full Chromium asks for it, the headless shell does not): no content, no 404
      res.writeHead(204, headers);
      res.end();
      return;
    }
    let target = file;
    try {
      if (target && fs.statSync(target).isDirectory()) target = path.join(target, 'index.html');
    } catch { /* not found below */ }
    let st;
    try {
      st = target && fs.statSync(target);
    } catch {
      st = null;
    }
    if (!st || !st.isFile()) {
      res.writeHead(404, { ...headers, 'Content-Type': 'text/plain' });
      res.end('not found');
      return;
    }
    headers['Content-Type'] = TYPES[path.extname(target).toLowerCase()] || 'application/octet-stream';
    headers['Accept-Ranges'] = 'bytes';
    let start = 0;
    let end = st.size - 1;
    let status = 200;
    const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || '');
    if (range && (range[1] || range[2])) {
      if (range[1]) {
        start = Number(range[1]);
        if (range[2]) end = Math.min(end, Number(range[2]));
      } else {
        start = Math.max(0, st.size - Number(range[2]));
      }
      if (start > end || start >= st.size) {
        res.writeHead(416, { ...headers, 'Content-Range': `bytes */${st.size}` });
        res.end();
        return;
      }
      status = 206;
      headers['Content-Range'] = `bytes ${start}-${end}/${st.size}`;
    }
    headers['Content-Length'] = end - start + 1;
    res.writeHead(status, headers);
    if (req.method === 'HEAD') {
      res.end();
      return;
    }
    const stream = fs.createReadStream(target, { start, end });
    stream.on('data', (chunk) => { stats.bytes += chunk.length; });
    stream.pipe(res);
  });

  return new Promise((resolveListen) => {
    server.listen(port, host, () => {
      const actual = server.address().port;
      if (!quiet) {
        console.log(`CzechSpeech dev server: http://localhost:${actual}/examples/basic/` +
          `${isolation ? '' : ' (no cross-origin isolation)'}`);
        for (const m of mounts) console.log(`  ${m.prefix} -> ${m.file || m.dir}`);
      }
      resolveListen({ server, port: actual, stats, close: () => new Promise((r) => server.close(r)) });
    });
  });
}

/** `rel` resolved inside `base`, or null when it would leave it. */
function inside(base, rel) {
  const target = path.resolve(base, rel.replace(/\\/g, '/'));
  return target === base || target.startsWith(base + path.sep) ? target : null;
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const args = process.argv.slice(2);
  const opt = (name) => {
    const i = args.indexOf(name);
    return i >= 0 ? args[i + 1] : null;
  };
  await startServer({
    port: Number(opt('--port') || 8790),
    model: opt('--model'),
    fixtures: opt('--fixtures'),
    isolation: !args.includes('--no-isolation'),
  });
}
