// Worker of the browser tests: what WebGPU looks like from inside a dedicated worker (where the recogniser runs).
// Reports the SDK's own decision (src/backend.js probeWebGPU) and the adapter's identity and features; no content.
import { probeWebGPU } from '../../src/backend.js';

self.onmessage = async () => {
  const out = { hasGpu: !!(self.navigator && self.navigator.gpu), probe: null, info: null, features: null, limits: null };
  try {
    out.probe = await probeWebGPU({ gpu: self.navigator && self.navigator.gpu });
    if (self.navigator.gpu) {
      const adapter = await self.navigator.gpu.requestAdapter();
      if (adapter) {
        const i = adapter.info || {};
        out.info = { vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description,
          isFallbackAdapter: !!i.isFallbackAdapter, subgroupMinSize: i.subgroupMinSize, subgroupMaxSize: i.subgroupMaxSize };
        out.features = [...adapter.features].sort();
        out.limits = { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize };
      }
    }
  } catch (e) {
    out.error = String(e && e.message ? e.message : e);
  }
  self.postMessage(out);
};
