// An in-memory stand-in for the browser's CacheStorage (Node.js has none), with the subset of the API model.js uses.
export class MemoryCacheStorage {
  constructor() {
    this.map = new Map();
  }

  async open(name) {
    if (!this.map.has(name)) this.map.set(name, new MemoryCache());
    return this.map.get(name);
  }
}

const urlOf = (req) => (typeof req === 'string' ? req : req.url);

export class MemoryCache {
  constructor() {
    this.entries = new Map();
    this.quotaBytes = Infinity;
  }

  get bytes() {
    let n = 0;
    for (const e of this.entries.values()) n += e.body.length;
    return n;
  }

  async match(req) {
    const e = this.entries.get(urlOf(req));
    return e ? new Response(e.body.slice(), { headers: e.headers }) : undefined;
  }

  async put(req, res) {
    const body = new Uint8Array(await res.arrayBuffer());
    if (this.bytes + body.length > this.quotaBytes) {
      throw new DOMException('quota exceeded', 'QuotaExceededError');
    }
    this.entries.set(urlOf(req), { body, headers: [...res.headers] });
  }

  async delete(req) {
    return this.entries.delete(urlOf(req));
  }

  async keys() {
    return [...this.entries.keys()].map((u) => new Request(u));
  }
}
