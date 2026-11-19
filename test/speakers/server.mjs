// A local static server for the speaker browser tests: the SDK sources under /src/ and exactly the files listed in
// `files` (URL path -> local file); everything else is 404. COOP/COEP/CORP headers only when `isolated`. Range requests
// are supported. It binds to 127.0.0.1 only.
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.wav': 'audio/wav',
  '.f32': 'application/octet-stream', '.txt': 'text/plain; charset=utf-8',
};

export async function startSpeakerServer({ files = {}, isolated = true } = {}) {
  const resolve = (pathname) => {
    if (Object.hasOwn(files, pathname)) return path.resolve(files[pathname]);
    if (pathname.startsWith('/src/')) {
      const p = path.resolve(ROOT, pathname.slice(1));
      return p.startsWith(path.join(ROOT, 'src') + path.sep) ? p : null;
    }
    return null;
  };
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    const headers = { 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' };
    if (isolated) {
      headers['Cross-Origin-Opener-Policy'] = 'same-origin';
      headers['Cross-Origin-Embedder-Policy'] = 'require-corp';
      headers['Cross-Origin-Resource-Policy'] = 'same-origin';
    }
    const file = req.method === 'GET' || req.method === 'HEAD' ? resolve(decodeURIComponent(url.pathname)) : null;
    let st;
    try { st = file && fs.statSync(file); } catch { st = null; }
    if (!st || !st.isFile()) {
      res.writeHead(url.pathname === '/favicon.ico' ? 204 : 404, headers);
      res.end();
      return;
    }
    const type = TYPES[path.extname(file)] || 'application/octet-stream';
    const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range || '');
    let start = 0;
    let end = st.size - 1;
    let status = 200;
    if (range && (range[1] || range[2])) {
      if (range[1]) { start = Number(range[1]); if (range[2]) end = Math.min(end, Number(range[2])); } else { start = Math.max(0, st.size - Number(range[2])); }
      if (start > end || start >= st.size) { res.writeHead(416, { ...headers, 'Content-Range': `bytes */${st.size}` }); res.end(); return; }
      status = 206;
      headers['Content-Range'] = `bytes ${start}-${end}/${st.size}`;
    }
    res.writeHead(status, { ...headers, 'Content-Type': type, 'Content-Length': end - start + 1, 'Accept-Ranges': 'bytes' });
    if (req.method === 'HEAD') { res.end(); return; }
    fs.createReadStream(file, { start, end }).pipe(res);
  });
  await new Promise((resolveListen) => server.listen(0, '127.0.0.1', resolveListen));
  const origin = `http://127.0.0.1:${server.address().port}`;
  return { origin, close: () => new Promise((r) => { server.closeAllConnections?.(); server.close(r); }) };
}
