// Approximate greedy coreset subsampling, as in
// patchcore.sampler.ApproximateGreedyCoresetSampler:
//   1. project features to 128-d with a random linear map
//   2. anchor distance of every point = mean distance to 10 random start points
//   3. repeat: pick argmax anchor distance, then anchor = min(anchor, dist to pick)
// The loop runs entirely on the GPU when WebGPU is available, otherwise on the
// CPU in Web Workers (several sharing memory when the page is cross-origin
// isolated, else one).

export const PROJECTION_DIM = 128;
const START_POINTS = 10;

/** Deterministic PRNG (mulberry32) returning floats in [0, 1). */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** [dim, PROJECTION_DIM] matrix with torch.nn.Linear's default init range. */
export function projectionMatrix(dim, seed) {
  const rand = rng(seed ^ 0x9e3779b9);
  const bound = 1 / Math.sqrt(dim);
  const w = new Float32Array(dim * PROJECTION_DIM);
  for (let i = 0; i < w.length; i++) w[i] = (rand() * 2 - 1) * bound;
  return w;
}

function pickStartPoints(n, seed) {
  const rand = rng(seed);
  const picks = new Set();
  const k = Math.min(START_POINTS, n);
  while (picks.size < k) picks.add(Math.floor(rand() * n));
  return [...picks];
}

/** Mean Euclidean distance from every point to the start points. */
export function initialAnchorDistances(feats, n, d, starts) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const bi = i * d;
    let total = 0;
    for (const s of starts) {
      const bs = s * d;
      let acc = 0;
      for (let k = 0; k < d; k++) {
        const t = feats[bi + k] - feats[bs + k];
        acc += t * t;
      }
      total += Math.sqrt(acc);
    }
    out[i] = total / starts.length;
  }
  return out;
}

/** minDist[i] = min(minDist[i], |x_i - x_s|) for i in [lo, hi). d must be a multiple of 4. */
export function updateRange(feats, minDist, lo, hi, d, s) {
  const bs = s * d;
  for (let i = lo; i < hi; i++) {
    if (minDist[i] === 0) continue;
    const base = i * d;
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (let k = 0; k < d; k += 4) {
      const t0 = feats[base + k] - feats[bs + k];
      const t1 = feats[base + k + 1] - feats[bs + k + 1];
      const t2 = feats[base + k + 2] - feats[bs + k + 2];
      const t3 = feats[base + k + 3] - feats[bs + k + 3];
      a0 += t0 * t0; a1 += t1 * t1; a2 += t2 * t2; a3 += t3 * t3;
    }
    const dist = Math.sqrt(a0 + a1 + a2 + a3);
    if (dist < minDist[i]) minDist[i] = dist;
  }
}

/** First index of the maximum of minDist over [lo, hi), like torch.argmax. */
export function argmaxRange(minDist, lo, hi) {
  let best = -1, bi = lo;
  for (let i = lo; i < hi; i++) if (minDist[i] > best) { best = minDist[i]; bi = i; }
  return [best, bi];
}

/** Single-threaded greedy loop (worker fallback without SharedArrayBuffer). Mutates minDist. */
export function greedyCpu(feats, n, d, count, minDist, onProgress) {
  const selected = new Int32Array(count);
  for (let it = 0; it < count; it++) {
    const [, s] = argmaxRange(minDist, 0, n);
    selected[it] = s;
    updateRange(feats, minDist, 0, n, d, s);
    if (onProgress && (it % 32 === 0 || it === count - 1)) onProgress(it + 1, count);
  }
  return selected;
}

const WGSL = /* wgsl */ `
struct Params { n: u32, d4: u32, groups: u32, total: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> feats: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> minDist: array<f32>;
@group(0) @binding(3) var<storage, read_write> partVal: array<f32>;
@group(0) @binding(4) var<storage, read_write> partIdx: array<u32>;
@group(0) @binding(5) var<storage, read_write> state: array<u32>;   // [iteration, current pick]
@group(0) @binding(6) var<storage, read_write> selected: array<u32>;

var<workgroup> sVal: array<f32, 256>;
var<workgroup> sIdx: array<u32, 256>;
var<workgroup> anchor: array<vec4f, 64>;

// argmax with ties going to the lower index, like torch.argmax
fn better(v: f32, i: u32, bv: f32, bi: u32) -> bool {
  return v > bv || (v == bv && i < bi);
}

fn reduceWorkgroup(lid: u32) {
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid < s) {
      let ov = sVal[lid + s];
      let oi = sIdx[lid + s];
      if (better(ov, oi, sVal[lid], sIdx[lid])) { sVal[lid] = ov; sIdx[lid] = oi; }
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(256)
fn argmaxPartial(@builtin(local_invocation_id) l: vec3u, @builtin(workgroup_id) w: vec3u) {
  var bv = -1.0;
  var bi = 0xffffffffu;
  let stride = P.groups * 256u;
  for (var i = w.x * 256u + l.x; i < P.n; i += stride) {
    let v = minDist[i];
    if (better(v, i, bv, bi)) { bv = v; bi = i; }
  }
  sVal[l.x] = bv;
  sIdx[l.x] = bi;
  workgroupBarrier();
  reduceWorkgroup(l.x);
  if (l.x == 0u) { partVal[w.x] = sVal[0]; partIdx[w.x] = sIdx[0]; }
}

@compute @workgroup_size(256)
fn argmaxFinal(@builtin(local_invocation_id) l: vec3u) {
  var bv = -1.0;
  var bi = 0xffffffffu;
  for (var i = l.x; i < P.groups; i += 256u) {
    if (better(partVal[i], partIdx[i], bv, bi)) { bv = partVal[i]; bi = partIdx[i]; }
  }
  sVal[l.x] = bv;
  sIdx[l.x] = bi;
  workgroupBarrier();
  reduceWorkgroup(l.x);
  if (l.x == 0u) {
    let it = state[0];
    if (it < P.total) { selected[it] = sIdx[0]; }
    state[0] = it + 1u;
    state[1] = sIdx[0];
  }
}

@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_id) l: vec3u,
          @builtin(num_workgroups) nw: vec3u) {
  let s = state[1];
  if (l.x < P.d4) { anchor[l.x] = feats[s * P.d4 + l.x]; }
  workgroupBarrier();
  let i = g.x + g.y * nw.x * 256u;
  if (i >= P.n) { return; }
  var acc = 0.0;
  let base = i * P.d4;
  for (var k = 0u; k < P.d4; k++) {
    let t = feats[base + k] - anchor[k];
    acc += dot(t, t);
  }
  minDist[i] = min(minDist[i], sqrt(acc));
}
`;

async function greedyGpu(device, feats, n, d, count, minDist, onProgress, signal) {
  const groups = Math.min(256, Math.ceil(n / 256));
  const U = GPUBufferUsage;
  const mk = (size, usage) => device.createBuffer({ size: Math.max(16, size), usage });
  const params = mk(16, U.UNIFORM | U.COPY_DST);
  const featBuf = mk(feats.byteLength, U.STORAGE | U.COPY_DST);
  const distBuf = mk(n * 4, U.STORAGE | U.COPY_DST);
  const partVal = mk(groups * 4, U.STORAGE);
  const partIdx = mk(groups * 4, U.STORAGE);
  const state = mk(8, U.STORAGE | U.COPY_DST);
  const selBuf = mk(count * 4, U.STORAGE | U.COPY_SRC);
  const readBuf = mk(count * 4, U.MAP_READ | U.COPY_DST);
  const buffers = [params, featBuf, distBuf, partVal, partIdx, state, selBuf, readBuf];
  try {
    device.queue.writeBuffer(params, 0, new Uint32Array([n, d / 4, groups, count]));
    device.queue.writeBuffer(featBuf, 0, feats);
    device.queue.writeBuffer(distBuf, 0, minDist);
    device.queue.writeBuffer(state, 0, new Uint32Array([0, 0]));

    const storage = (binding, type = 'storage') => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const layout = device.createBindGroupLayout({
      entries: [
        storage(0, 'uniform'), storage(1, 'read-only-storage'), storage(2), storage(3),
        storage(4), storage(5), storage(6),
      ],
    });
    const module = device.createShaderModule({ code: WGSL });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const [partial, final, update] = await Promise.all(
      ['argmaxPartial', 'argmaxFinal', 'update'].map((entryPoint) => device.createComputePipelineAsync({
        layout: pipelineLayout, compute: { module, entryPoint },
      })),
    );
    const bindGroup = device.createBindGroup({
      layout,
      entries: [params, featBuf, distBuf, partVal, partIdx, state, selBuf].map((buffer, binding) => ({
        binding, resource: { buffer },
      })),
    });
    const updateGroups = Math.ceil(n / 256);
    const maxX = device.limits.maxComputeWorkgroupsPerDimension;
    const ux = Math.min(updateGroups, maxX);
    const uy = Math.ceil(updateGroups / ux);

    const BATCH = 256;
    for (let done = 0; done < count; ) {
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      const steps = Math.min(BATCH, count - done);
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, bindGroup);
      for (let s = 0; s < steps; s++) {
        pass.setPipeline(partial);
        pass.dispatchWorkgroups(groups);
        pass.setPipeline(final);
        pass.dispatchWorkgroups(1);
        pass.setPipeline(update);
        pass.dispatchWorkgroups(ux, uy);
      }
      pass.end();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      done += steps;
      onProgress?.(done, count);
    }
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(selBuf, 0, readBuf, 0, count * 4);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const out = new Int32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
    return out;
  } finally {
    for (const b of buffers) b.destroy();
  }
}

const workerUrl = () => new URL('./coreset-worker.js', import.meta.url);

function greedyInWorker(feats, n, d, count, minDist, onProgress, signal) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(workerUrl(), { type: 'module' });
    const stop = () => { worker.terminate(); reject(new DOMException('Aborted', 'AbortError')); };
    signal?.addEventListener('abort', stop, { once: true });
    worker.onmessage = ({ data }) => {
      if (data.progress) onProgress?.(data.progress, count);
      if (data.selected) {
        signal?.removeEventListener('abort', stop);
        worker.terminate();
        resolve(data.selected);
      }
    };
    worker.onerror = (e) => { worker.terminate(); reject(new Error(e.message || 'coreset worker failed')); };
    worker.postMessage({ mode: 'single', feats, n, d, count, minDist }, [feats.buffer, minDist.buffer]);
  });
}

/**
 * Multi-worker CPU loop over SharedArrayBuffers (needs a cross-origin isolated
 * page). Each worker owns a slice of the points; per iteration they publish
 * their local argmax, meet at one barrier, all reduce the same global argmax,
 * and update their own slice. Same picks as the single-threaded loop.
 */
function greedyInWorkers(feats, n, d, count, minDist, workers, onProgress, signal) {
  const shared = (Ctor, length) => new Ctor(new SharedArrayBuffer(length * Ctor.BYTES_PER_ELEMENT));
  const buffers = {
    feats: shared(Float32Array, feats.length),
    minDist: shared(Float32Array, n),
    bestVal: shared(Float32Array, 2 * workers),
    bestIdx: shared(Int32Array, 2 * workers),
    ctrl: shared(Int32Array, 2),
    selected: shared(Int32Array, count),
  };
  buffers.feats.set(feats);
  buffers.minDist.set(minDist);
  return new Promise((resolve, reject) => {
    const pool = [];
    let finished = 0;
    const cleanup = () => { for (const w of pool) w.terminate(); };
    const stop = () => { cleanup(); reject(new DOMException('Aborted', 'AbortError')); };
    signal?.addEventListener('abort', stop, { once: true });
    for (let w = 0; w < workers; w++) {
      const worker = new Worker(workerUrl(), { type: 'module' });
      worker.onmessage = ({ data }) => {
        if (data.progress) onProgress?.(data.progress, count);
        if (data.done && ++finished === workers) {
          signal?.removeEventListener('abort', stop);
          cleanup();
          resolve(Int32Array.from(buffers.selected));
        }
      };
      worker.onerror = (e) => { cleanup(); reject(new Error(e.message || 'coreset worker failed')); };
      worker.postMessage({ mode: 'shared', w, workers, n, d, count, ...buffers });
      pool.push(worker);
    }
  });
}

/**
 * projected: Float32Array n x d (d = PROJECTION_DIM). Returns Int32Array of
 * selected row indices in selection order. `feats` is transferred to the
 * worker on the CPU path, so callers must not reuse it afterwards.
 */
export async function approximateGreedyCoreset(projected, n, d, percentage, { device, seed = 0, onProgress, signal } = {}) {
  const count = Math.floor(n * percentage);
  if (count <= 0) throw new Error('Coreset is empty: add more images or raise the percentage');
  if (percentage >= 1) return Int32Array.from({ length: n }, (_, i) => i);
  const minDist = initialAnchorDistances(projected, n, d, pickStartPoints(n, seed));
  const fits = device && d % 4 === 0 && d <= 256 && projected.byteLength <= device.limits.maxStorageBufferBindingSize;
  if (fits) return greedyGpu(device, projected, n, d, count, minDist, onProgress, signal);
  const threads = Math.min(12, navigator.hardwareConcurrency || 1);
  if (self.crossOriginIsolated && threads > 1) {
    return greedyInWorkers(projected, n, d, count, minDist, threads, onProgress, signal);
  }
  return greedyInWorker(projected, n, d, count, minDist, onProgress, signal);
}
