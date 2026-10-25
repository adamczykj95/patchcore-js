// Exact nearest-neighbour search against the memory bank (faiss IndexFlatL2, k=1)
// using the tiny knn.onnx graph. The bank is split into chunks to bound the
// distance-matrix size; on WebGPU the chunks live on the GPU for the lifetime
// of the bank, and the query features come straight from the embed session's
// GPU output, so a query only downloads Q floats.
import { ort } from './runtime.js';

const CHUNK_ROWS = 16384;

export class NearestNeighbor {
  constructor(rt, session) {
    this.rt = rt;
    this.session = session;
    this.chunks = [];
    this.count = 0;
  }

  setBank(features, count, dim) {
    this.clear();
    const device = this.rt.device;
    for (let start = 0; start < count; start += CHUNK_ROWS) {
      const rows = Math.min(CHUNK_ROWS, count - start);
      const view = features.subarray(start * dim, (start + rows) * dim);
      if (this.rt.backend === 'webgpu') {
        const buffer = device.createBuffer({
          size: view.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, view);
        this.chunks.push(ort.Tensor.fromGpuBuffer(buffer, {
          dataType: 'float32', dims: [rows, dim], dispose: () => buffer.destroy(),
        }));
      } else {
        this.chunks.push(new ort.Tensor('float32', view, [rows, dim]));
      }
    }
    this.count = count;
  }

  /** query: ort.Tensor [Q, D] (CPU or GPU). Returns Float32Array of Q squared distances. */
  async query(query) {
    if (!this.chunks.length) throw new Error('No memory bank loaded');
    let best = null;
    for (const bank of this.chunks) {
      const { distance } = await this.session.run({ query, bank });
      const d = distance.data;
      if (!best) best = Float32Array.from(d);
      else for (let i = 0; i < best.length; i++) if (d[i] < best[i]) best[i] = d[i];
      distance.dispose();
    }
    return best;
  }

  clear() {
    for (const t of this.chunks) t.dispose();
    this.chunks = [];
    this.count = 0;
  }
}
