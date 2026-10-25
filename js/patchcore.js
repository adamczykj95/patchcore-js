// PatchCore in the browser: fit a memory bank from good images, or load one,
// and score test images. Mirrors patchcore.patchcore.PatchCore.
import { ort, createRuntime } from './runtime.js';
import { preprocess } from './preprocess.js';
import { buildResult } from './scoring.js';
import { NearestNeighbor } from './knn.js';
import { approximateGreedyCoreset, projectionMatrix, PROJECTION_DIM } from './coreset.js';
import { checkCompatible } from './bank.js';

const now = () => performance.now();

export class PatchCore {
  /**
   * backend: 'auto' | 'webgpu' | 'wasm'; precision: 'fp32' | 'fp16' (fp16 needs WebGPU).
   * onStatus(message, fraction?) reports model download progress.
   */
  static async create({ backend = 'auto', modelId, precision = 'fp32', modelsUrl = 'models/', onStatus } = {}) {
    const rt = await createRuntime({ backend, modelsUrl });
    const cfg = rt.manifest.models.find((m) => m.id === modelId) || rt.manifest.models[0];
    const useFp16 = precision === 'fp16' && rt.backend === 'webgpu' && rt.f16 && cfg.files.fp16;
    const entry = useFp16 ? cfg.files.fp16 : cfg.files.fp32;
    const mb = (b) => (b / 1e6).toFixed(0);
    const embed = await rt.createSession(entry, {
      gpuOutput: true,
      onProgress: (loaded, total) => onStatus?.(`Downloading model ${mb(loaded)} / ${mb(total || entry.bytes)} MB`, loaded / (total || entry.bytes)),
    });
    onStatus?.('Compiling model…');
    const [knn, matmul] = await Promise.all([rt.createSession(rt.manifest.knn), rt.createSession(rt.manifest.matmul)]);
    const pc = new PatchCore(rt, { ...cfg, precision: useFp16 ? 'fp16' : 'fp32' }, embed, knn, matmul);
    // warm-up so the first real image doesn't pay for shader compilation
    const warm = await pc.embed(new Float32Array(3 * cfg.imagesize * cfg.imagesize));
    warm.dispose();
    return pc;
  }

  constructor(rt, cfg, embedSession, knnSession, matmulSession) {
    this.rt = rt;
    this.cfg = cfg;
    this.embedSession = embedSession;
    this.matmulSession = matmulSession;
    this.knn = new NearestNeighbor(rt, knnSession);
    this.bank = null;
    this.patches = cfg.grid[0] * cfg.grid[1];
  }

  get backend() { return this.rt.backend; }

  /** Model input [3,S,S] -> ort.Tensor [1, P, D] (on GPU under WebGPU). Caller disposes. */
  async embed(tensor) {
    const s = this.cfg.imagesize;
    const image = new ort.Tensor('float32', tensor, [1, 3, s, s]);
    const { features } = await this.embedSession.run({ image });
    return features;
  }

  /** View a [1,P,D] tensor as [P,D] without copying. */
  asMatrix(t) {
    const dims = [this.patches, this.cfg.dim];
    if (t.location === 'gpu-buffer') return ort.Tensor.fromGpuBuffer(t.gpuBuffer, { dataType: 'float32', dims });
    return new ort.Tensor('float32', t.data, dims);
  }

  async features(source) {
    const prep = await preprocess(source, this.cfg);
    const t = await this.embed(prep.tensor);
    const data = await t.getData(true);
    return { data, prep };
  }

  /**
   * Build a memory bank from good images.
   * opts: coreset (fraction), cacheLimitMB (keep full features in RAM below this,
   * otherwise re-embed for the final gather), seed, name, onProgress(phase, done, total), signal.
   */
  async fit(sources, { coreset = 0.1, cacheLimitMB = 768, seed = 0, name = 'bank', onProgress, signal } = {}) {
    const { dim } = this.cfg;
    const P = this.patches;
    const n = sources.length * P;
    const keepFull = (n * dim * 4) / 2 ** 20 <= cacheLimitMB;
    const W = new ort.Tensor('float32', projectionMatrix(dim, seed), [dim, PROJECTION_DIM]);
    const projected = new Float32Array(n * PROJECTION_DIM);
    const cached = [];
    const t0 = now();

    const check = () => { if (signal?.aborted) throw new DOMException('Aborted', 'AbortError'); };
    for (let i = 0; i < sources.length; i++) {
      check();
      const prep = await preprocess(sources[i], this.cfg);
      const out = await this.embed(prep.tensor);
      const { c } = await this.matmulSession.run({ a: this.asMatrix(out), b: W });
      projected.set(await c.getData(true), i * P * PROJECTION_DIM);
      if (keepFull) cached.push(await out.getData(true));
      else out.dispose();
      onProgress?.('embed', i + 1, sources.length);
    }
    const tEmbed = now();

    const selected = await approximateGreedyCoreset(projected, n, PROJECTION_DIM, coreset, {
      device: this.rt.device, seed, signal, onProgress: (d, t) => onProgress?.('coreset', d, t),
    });
    const tCoreset = now();

    const bank = new Float32Array(selected.length * dim);
    const byImage = new Map();
    selected.forEach((row, j) => {
      const img = Math.floor(row / P);
      if (!byImage.has(img)) byImage.set(img, []);
      byImage.get(img).push([row % P, j]);
    });
    let gathered = 0;
    for (const [img, picks] of byImage) {
      check();
      const feats = keepFull ? cached[img] : (await this.features(sources[img])).data;
      for (const [patch, j] of picks) bank.set(feats.subarray(patch * dim, (patch + 1) * dim), j * dim);
      onProgress?.('gather', ++gathered, byImage.size);
    }
    W.dispose();

    const header = {
      name,
      count: selected.length,
      dim,
      model: {
        backbone: this.cfg.backbone, layers: this.cfg.layers, imagesize: this.cfg.imagesize,
        resize: this.cfg.resize, patchsize: this.cfg.patchsize, dim,
      },
      num_nn: 1,
      source: 'patchcore-js',
      coreset: { method: 'approx_greedy', percentage: coreset, seed, images: sources.length },
      created: new Date().toISOString(),
      timings_ms: {
        embed: Math.round(tEmbed - t0), coreset: Math.round(tCoreset - tEmbed), gather: Math.round(now() - tCoreset),
      },
      threshold: null,
    };
    this.setBank({ features: bank, header });
    return this.bank;
  }

  setBank({ features, header }) {
    checkCompatible(header, this.cfg);
    if ((header.num_nn ?? 1) !== 1) {
      console.warn(`bank was trained with anomaly_scorer_num_nn=${header.num_nn}; scoring uses k=1`);
    }
    this.knn.setBank(features, header.count, header.dim);
    this.bank = { features, header };
  }

  /** Score one image. Returns { score, map, patchScores, rgb, size, timings }. */
  async predict(source) {
    if (!this.bank) throw new Error('Fit or load a memory bank first');
    const t0 = now();
    const prep = await preprocess(source, this.cfg);
    const t1 = now();
    const out = await this.embed(prep.tensor);
    const t2 = now();
    const dist = await this.knn.query(this.asMatrix(out));
    out.dispose();
    const t3 = now();
    const result = buildResult(dist, this.cfg.grid, this.cfg.imagesize);
    const t4 = now();
    return {
      ...result,
      rgb: prep.rgb,
      timings: { preprocess: t1 - t0, embed: t2 - t1, knn: t3 - t2, post: t4 - t3, total: t4 - t0 },
    };
  }

  dispose() {
    this.knn.clear();
    this.embedSession.release();
    this.matmulSession.release();
    this.knn.session.release();
  }
}
