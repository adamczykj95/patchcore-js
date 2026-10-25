// ONNX Runtime Web setup: backend choice, a shared WebGPU device, cached model
// downloads and session creation.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.webgpu.min.mjs';

export { ort };

const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
const MODEL_CACHE = 'patchcore-models-v1';

async function probeWebGpu() {
  if (!('gpu' in navigator)) return null;
  let adapter = null;
  // requestAdapter can transiently return null while the GPU process starts up
  for (let attempt = 0; attempt < 3 && !adapter; attempt++) {
    if (attempt) await new Promise((r) => setTimeout(r, 300 * attempt));
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  }
  if (!adapter) return null;
  const info = adapter.info || {};
  return {
    label: [info.vendor, info.architecture || info.description].filter(Boolean).join(' ') || 'WebGPU',
    f16: adapter.features.has('shader-f16'),
  };
}

/** Fetch with download progress, cached in Cache Storage keyed by content hash. */
export async function fetchCached(url, hash, onProgress) {
  const key = hash ? `${new URL(url, location.href).href}?v=${hash}` : url;
  let cache = null;
  try {
    cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(key);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch {
    cache = null;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.(loaded, total);
  }
  const buf = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  if (cache) {
    try {
      // drop stale versions of the same file before storing the new one
      for (const req of await cache.keys()) if (req.url.split('?')[0] === key.split('?')[0]) await cache.delete(req);
      await cache.put(key, new Response(buf));
    } catch { /* quota or private mode: just skip caching */ }
  }
  return buf;
}

/**
 * Create the runtime. backend: 'auto' | 'webgpu' | 'wasm'.
 * Returns { backend, device, deviceLabel, f16, threads, manifest, createSession }.
 */
export async function createRuntime({ backend = 'auto', modelsUrl = 'models/' } = {}) {
  ort.env.wasm.wasmPaths = ORT_DIST;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
  ort.env.logLevel = 'error';

  let gpu = null;
  if (backend !== 'wasm') {
    try {
      gpu = await probeWebGpu();
    } catch (err) {
      console.warn('WebGPU unavailable:', err);
    }
    if (!gpu && backend === 'webgpu') throw new Error('WebGPU is not available in this browser');
  }
  const chosen = gpu ? 'webgpu' : 'wasm';

  const manifest = await (await fetch(new URL('manifest.json', new URL(modelsUrl, location.href)), { cache: 'no-cache' })).json();

  async function createSession(entry, { onProgress, gpuOutput = false } = {}) {
    const url = new URL(entry.path, new URL(modelsUrl, location.href)).href;
    const bytes = await fetchCached(url, entry.sha256, onProgress);
    const opts = { executionProviders: [chosen], graphOptimizationLevel: 'all' };
    if (chosen === 'webgpu' && gpuOutput) opts.preferredOutputLocation = 'gpu-buffer';
    const session = await ort.InferenceSession.create(bytes, opts);
    // ORT owns the GPUDevice; GPU tensors and our own kernels must use the same one.
    if (chosen === 'webgpu' && !runtime.device) runtime.device = await ort.env.webgpu.device;
    return session;
  }

  const runtime = {
    backend: chosen,
    device: null, // set once the first WebGPU session exists
    deviceLabel: gpu ? gpu.label : `WASM ×${ort.env.wasm.numThreads}`,
    f16: Boolean(gpu?.f16),
    threads: ort.env.wasm.numThreads,
    manifest,
    createSession,
  };
  return runtime;
}
