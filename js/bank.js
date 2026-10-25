// .pcbank memory-bank files (same layout as tools/pcbank.py) and an IndexedDB shelf.
//   8 bytes  "PCBANK1\0"
//   uint32   header length H (little endian)
//   H bytes  JSON header, space-padded so float data is 4-byte aligned
//   float32  count x dim rows

const MAGIC = 'PCBANK1\0';

export function encodeBank(features, header) {
  const full = { ...header, format: 1, count: header.count, dim: header.dim };
  let json = new TextEncoder().encode(JSON.stringify(full));
  const pad = (4 - ((8 + 4 + json.length) % 4)) % 4;
  if (pad) {
    const padded = new Uint8Array(json.length + pad).fill(0x20);
    padded.set(json);
    json = padded;
  }
  const out = new Uint8Array(12 + json.length + features.byteLength);
  out.set(new TextEncoder().encode(MAGIC), 0);
  new DataView(out.buffer).setUint32(8, json.length, true);
  out.set(json, 12);
  out.set(new Uint8Array(features.buffer, features.byteOffset, features.byteLength), 12 + json.length);
  return out.buffer;
}

export function decodeBank(buffer) {
  const bytes = new Uint8Array(buffer);
  if (new TextDecoder().decode(bytes.subarray(0, 8)) !== MAGIC) {
    throw new Error('Not a .pcbank file');
  }
  const hlen = new DataView(buffer).getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + hlen)));
  const features = new Float32Array(buffer, 12 + hlen, header.count * header.dim);
  return { header, features };
}

/** Throws if the bank was built with a different feature extractor than `cfg`. */
export function checkCompatible(header, cfg) {
  const m = header.model || {};
  const problems = [];
  if (m.backbone && m.backbone !== cfg.backbone) problems.push(`backbone ${m.backbone} ≠ ${cfg.backbone}`);
  if (m.layers && m.layers.join() !== cfg.layers.join()) problems.push(`layers ${m.layers} ≠ ${cfg.layers}`);
  for (const key of ['imagesize', 'resize', 'patchsize']) {
    if (m[key] !== undefined && m[key] !== cfg[key]) problems.push(`${key} ${m[key]} ≠ ${cfg[key]}`);
  }
  if (header.dim !== cfg.dim) problems.push(`dim ${header.dim} ≠ ${cfg.dim}`);
  if (problems.length) throw new Error(`Bank does not match the loaded model: ${problems.join(', ')}`);
}

// ---- IndexedDB shelf ------------------------------------------------------

const DB_NAME = 'patchcore-js';
const STORE = 'banks';

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: 'id' });
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx(mode, fn) {
  const db = await openDb();
  try {
    return await new Promise((resolve, reject) => {
      const t = db.transaction(STORE, mode);
      const req = fn(t.objectStore(STORE));
      t.oncomplete = () => resolve(req?.result);
      t.onerror = () => reject(t.error);
      t.onabort = () => reject(t.error);
    });
  } finally {
    db.close();
  }
}

export async function saveBank(buffer, header) {
  const id = header.id || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
  const { header: stored } = decodeBank(buffer);
  await tx('readwrite', (s) => s.put({
    id,
    name: stored.name || 'bank',
    count: stored.count,
    dim: stored.dim,
    created: stored.created,
    threshold: stored.threshold ?? null,
    bytes: buffer.byteLength,
    buffer,
  }));
  return id;
}

export async function listBanks() {
  const rows = (await tx('readonly', (s) => s.getAll())) || [];
  return rows
    .map(({ buffer, ...meta }) => meta)
    .sort((a, b) => String(b.created).localeCompare(String(a.created)));
}

export async function loadBank(id) {
  const row = await tx('readonly', (s) => s.get(id));
  if (!row) throw new Error('Bank not found');
  return row.buffer;
}

export function deleteBank(id) {
  return tx('readwrite', (s) => s.delete(id));
}
