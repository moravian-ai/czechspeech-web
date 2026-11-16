// Counts the WebGPU buffers this worker's devices create and destroy (imported first by gpu-bytes-worker.js, before the
// SDK's worker entry), independently of the SDK's own tracker, which reports its total to the page only when the model is
// loaded. Asked on the BroadcastChannel 'czechspeech-e2e-gpu' with { type: 'check', id }, the recogniser worker answers
// { type: 'gpu', id, worker, liveBytes, liveBuffers, peakBytes, devices }. When a device is destroyed it posts
// { type: 'gpu-destroyed', worker, leftBytes, leftBuffers, peakBytes }: what was still alive at that moment (buffers
// the owner had not destroyed itself; the device frees them), then the count is zero.
const stats = { liveBytes: 0, liveBuffers: 0, peakBytes: 0, devices: 0 };
const channel = new BroadcastChannel('czechspeech-e2e-gpu');

function instrument(device) {
  stats.devices++;
  const own = new Map(); // buffer -> size, of this device
  const createBuffer = device.createBuffer.bind(device);
  device.createBuffer = (descriptor) => {
    const buffer = createBuffer(descriptor);
    const size = Number(descriptor && descriptor.size) || buffer.size || 0;
    own.set(buffer, size);
    stats.liveBytes += size;
    stats.liveBuffers++;
    stats.peakBytes = Math.max(stats.peakBytes, stats.liveBytes);
    const destroy = buffer.destroy.bind(buffer);
    buffer.destroy = () => {
      if (own.delete(buffer)) { stats.liveBytes -= size; stats.liveBuffers--; }
      destroy();
    };
    return buffer;
  };
  const destroyDevice = device.destroy.bind(device);
  device.destroy = () => {
    let leftBytes = 0;
    for (const s of own.values()) leftBytes += s;
    const leftBuffers = own.size;
    channel.postMessage({ type: 'gpu-destroyed', worker: self.name, leftBytes, leftBuffers, peakBytes: stats.peakBytes });
    stats.liveBytes -= leftBytes;
    stats.liveBuffers -= leftBuffers;
    own.clear();
    destroyDevice();
  };
}

if (typeof GPUAdapter !== 'undefined') {
  const requestDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function patched(...args) {
    const device = await requestDevice.apply(this, args);
    instrument(device);
    return device;
  };
}

channel.onmessage = (e) => {
  if (!e.data || e.data.type !== 'check' || self.name !== 'czechspeech-recognizer') return;
  channel.postMessage({ type: 'gpu', id: e.data.id, worker: self.name, ...stats });
};
