import { PatchCore } from './patchcore.js';
import { IMAGE_EXTENSIONS } from './preprocess.js';
import { auroc } from './scoring.js';
import { encodeBank, decodeBank, saveBank, listBanks, loadBank, deleteBank } from './bank.js';

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '–');
const mb = (bytes) => `${(bytes / 2 ** 20).toFixed(bytes < 10 * 2 ** 20 ? 1 : 0)} MB`;

// ---------------------------------------------------------------- state ----

const SETTINGS_KEY = 'patchcore-js.settings';
const defaults = { backend: 'auto', precision: 'fp32', modelId: '', cacheMB: 768, seed: 0 };
function loadSettings() {
  try { return { ...defaults, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}') }; } catch { return { ...defaults }; }
}
function storeSettings() {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings)); } catch { /* storage unavailable */ }
}

const state = {
  settings: loadSettings(),
  pc: null,
  fitFiles: [],
  bankId: null,
  bankImages: null,
  threshold: null,
  results: [],
  evalGood: [],
  evalBad: [],
  evalSuggestion: null,
  fitAbort: null,
};

// Every model call goes through one queue: ORT sessions do not allow overlapping runs.
let queue = Promise.resolve();
const exclusive = (fn) => {
  const run = queue.then(fn, fn);
  queue = run.catch(() => {});
  return run;
};

// ------------------------------------------------------------- helpers ----

function toast(message, kind = 'info', ms = 4200) {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = message;
  $('toasts').append(el);
  setTimeout(() => el.remove(), ms);
}

function setStatus(kind, text) {
  $('status').dataset.state = kind;
  $('status-text').textContent = text;
}

function fail(err) {
  if (err?.name === 'AbortError') return;
  console.error(err);
  toast(err?.message || String(err), 'error', 7000);
}

const fileName = (f) => f.webkitRelativePath || f.name;
const sortFiles = (files) => files.sort((a, b) => fileName(a).localeCompare(fileName(b), undefined, { numeric: true }));
const isImage = (f) => f.type.startsWith('image/') || IMAGE_EXTENSIONS.test(f.name);

async function filesFromDrop(dataTransfer) {
  const out = [];
  const readAll = (reader) => new Promise((resolve, reject) => {
    const entries = [];
    const next = () => reader.readEntries((batch) => {
      if (!batch.length) resolve(entries);
      else { entries.push(...batch); next(); }
    }, reject);
    next();
  });
  async function walk(entry, path) {
    if (entry.isFile) {
      const file = await new Promise((res, rej) => entry.file(res, rej));
      Object.defineProperty(file, 'webkitRelativePath', { value: path + file.name });
      out.push(file);
    } else if (entry.isDirectory) {
      for (const child of await readAll(entry.createReader())) await walk(child, `${path}${entry.name}/`);
    }
  }
  const entries = [...dataTransfer.items].map((i) => i.webkitGetAsEntry?.()).filter(Boolean);
  if (entries.length) for (const e of entries) await walk(e, '');
  else out.push(...dataTransfer.files);
  return sortFiles(out.filter(isImage));
}

/** Wire a dropzone plus its file/folder inputs to a callback receiving image Files. */
function wireDrop(zone, inputs, onFiles) {
  const deliver = (files) => { if (files.length) onFiles(files); else toast('No images found in that selection'); };
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('over'));
  zone.addEventListener('drop', async (e) => {
    e.preventDefault();
    zone.classList.remove('over');
    if (zone.dataset.disabled === 'true') return;
    deliver(await filesFromDrop(e.dataTransfer));
  });
  zone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); inputs[0]?.click(); }
  });
  for (const input of inputs) {
    input.addEventListener('change', () => {
      deliver(sortFiles([...input.files].filter(isImage)));
      input.value = '';
    });
  }
}

// ------------------------------------------------------------ colormap ----

// Turbo colormap (Google, Apache-2.0), polynomial fit by A. Mikhailov.
const LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    lut[i * 3] = 34.61 + t * (1172.33 - t * (10793.56 - t * (33300.12 - t * (38394.49 - t * 14825.05))));
    lut[i * 3 + 1] = 23.31 + t * (557.33 + t * (1225.33 - t * (3574.96 - t * (1073.77 + t * 707.56))));
    lut[i * 3 + 2] = 27.2 + t * (3211.1 - t * (15327.97 - t * (27814 - t * (22569.18 - t * 6838.66))));
  }
  return lut;
})();

function mapRange(result) {
  if ($('scale-mode').value === 'threshold' && state.threshold) {
    return { lo: state.threshold * 0.8, hi: state.threshold * 3, fade: true };
  }
  let lo = Infinity, hi = -Infinity;
  for (const v of result.map) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return { lo, hi: hi > lo ? hi : lo + 1e-6, fade: false };
}

function drawResult(canvas, result, withOverlay = true) {
  const { size, rgb, map } = result;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(size, size);
  const px = img.data;
  const opacity = withOverlay ? Number($('opacity').value) / 100 : 0;
  const { lo, hi, fade } = mapRange(result);
  const span = hi - lo;
  for (let i = 0, n = size * size; i < n; i++) {
    let r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
    if (opacity > 0) {
      const t = Math.min(1, Math.max(0, (map[i] - lo) / span));
      const a = fade ? opacity * Math.sqrt(t) : opacity;
      const c = Math.round(t * 255) * 3;
      r += (LUT[c] - r) * a;
      g += (LUT[c + 1] - g) * a;
      b += (LUT[c + 2] - b) * a;
    }
    px[i * 4] = r; px[i * 4 + 1] = g; px[i * 4 + 2] = b; px[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// -------------------------------------------------------------- engine ----

async function initEngine() {
  const s = state.settings;
  setStatus('loading', 'Loading model…');
  disableModelUi(true);
  const bank = state.pc?.bank;
  try {
    state.pc?.dispose();
  } catch { /* already gone */ }
  state.pc = null;
  try {
    state.pc = await PatchCore.create({
      backend: s.backend, precision: s.precision, modelId: s.modelId,
      onStatus: (msg) => setStatus('loading', msg),
    });
  } catch (err) {
    setStatus('error', 'Model failed to load');
    fail(err);
    return;
  }
  const pc = state.pc;
  if (bank) {
    try {
      pc.setBank(bank);
    } catch (err) {
      $('bank-card').hidden = true;
      $('threshold-card').hidden = true;
      fail(err);
    }
  }
  const sel = $('set-model');
  sel.innerHTML = '';
  for (const m of pc.rt.manifest.models) {
    sel.add(new Option(`${m.backbone} ${m.layers.join('+')} · ${m.imagesize}px · patch ${m.patchsize}`, m.id));
  }
  sel.value = pc.cfg.id;
  const where = pc.backend === 'webgpu' ? `WebGPU · ${pc.rt.deviceLabel}` : pc.rt.deviceLabel;
  setStatus('ready', `${where} · ${pc.cfg.precision}`);
  if (s.precision === 'fp16' && pc.cfg.precision !== 'fp16') {
    toast('fp16 needs WebGPU with shader-f16 support; using fp32 instead');
  }
  disableModelUi(false);
  updateFitNote();
}

function disableModelUi(disabled) {
  $('fit-btn').disabled = disabled || !state.fitFiles.length;
  $('eval-run').disabled = disabled || !state.pc?.bank || !state.evalGood.length || !state.evalBad.length;
  const inspectable = !disabled && Boolean(state.pc?.bank);
  $('inspect-drop').dataset.disabled = String(!inspectable);
  $('inspect-hint').textContent = inspectable ? 'Scored against the active memory bank' : 'Build or load a memory bank first';
}

// ----------------------------------------------------------------- fit ----

function setFitFiles(files) {
  state.fitFiles = files;
  $('fit-list').hidden = !files.length;
  $('fit-count').textContent = `${files.length} image${files.length === 1 ? '' : 's'}`;
  const thumbs = $('fit-thumbs');
  for (const img of thumbs.querySelectorAll('img')) URL.revokeObjectURL(img.src);
  thumbs.innerHTML = '';
  for (const f of files.slice(0, 16)) {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(f);
    img.alt = '';
    thumbs.append(img);
  }
  $('fit-btn').disabled = !files.length || !state.pc;
  updateFitNote();
}

function splitHoldout(files) {
  if (!$('holdout').checked || files.length < 10) return { train: files, holdout: [] };
  const holdout = files.filter((_, i) => i % 10 === 9);
  return { train: files.filter((_, i) => i % 10 !== 9), holdout };
}

function updateFitNote() {
  const pct = Number($('coreset').value);
  $('coreset-out').textContent = `${pct}%`;
  const pc = state.pc;
  if (!pc || !state.fitFiles.length) {
    $('coreset-note').textContent = 'Fraction of patch features kept. 10% is the paper default; 1% is faster with a small accuracy cost.';
    return;
  }
  const { train } = splitHoldout(state.fitFiles);
  const patches = train.length * pc.patches;
  const kept = Math.floor((patches * pct) / 100);
  let note = `${train.length} images → ${patches.toLocaleString()} patches → keep ${kept.toLocaleString()} (${mb(kept * pc.cfg.dim * 4)}).`;
  if (pc.backend !== 'webgpu' && kept > 3000) note += ' Without WebGPU the coreset step runs on the CPU and may take minutes; consider 1–2%.';
  $('coreset-note').textContent = note;
}

const PHASES = {
  embed: { label: 'Extracting features', from: 0, to: 0.5 },
  coreset: { label: 'Selecting coreset', from: 0.5, to: 0.95 },
  gather: { label: 'Collecting bank features', from: 0.95, to: 1 },
  calibrate: { label: 'Calibrating threshold', from: 1, to: 1 },
};

async function runFit() {
  const pc = state.pc;
  const { train, holdout } = splitHoldout(state.fitFiles);
  const first = fileName(state.fitFiles[0]);
  const name = first.includes('/') ? first.split('/')[0] : 'bank';
  const ac = new AbortController();
  state.fitAbort = ac;
  $('fit-btn').disabled = true;
  $('fit-cancel').hidden = false;
  $('fit-progress').hidden = false;
  let phaseStart = performance.now();
  let current = '';
  const progress = (phase, done, total) => {
    if (phase !== current) { current = phase; phaseStart = performance.now(); }
    const p = PHASES[phase];
    $('fit-phase').textContent = `${p.label} ${done.toLocaleString()} / ${total.toLocaleString()}`;
    const elapsed = (performance.now() - phaseStart) / 1000;
    const eta = done > 0 && done < total ? (elapsed / done) * (total - done) : 0;
    $('fit-eta').textContent = eta > 1 ? `~${Math.ceil(eta)} s left` : '';
    $('fit-bar').style.width = `${(p.from + (p.to - p.from) * (done / total)) * 100}%`;
  };
  try {
    await exclusive(async () => {
      const bank = await pc.fit(train, {
        coreset: Number($('coreset').value) / 100,
        cacheLimitMB: state.settings.cacheMB,
        seed: state.settings.seed,
        name,
        signal: ac.signal,
        onProgress: progress,
      });
      bank.header.images = train.length;
      let threshold = null;
      if (holdout.length) {
        const scores = [];
        for (let i = 0; i < holdout.length; i++) {
          if (ac.signal.aborted) throw new DOMException('Aborted', 'AbortError');
          scores.push((await pc.predict(holdout[i])).score);
          progress('calibrate', i + 1, holdout.length);
        }
        threshold = Math.max(...scores);
        bank.header.threshold = threshold;
        bank.header.threshold_note = `Max score of ${holdout.length} held-out good images`;
      }
      state.bankId = null;
      onBankActive('Fitted here');
      const t = bank.header.timings_ms;
      toast(`Memory bank ready: ${bank.header.count.toLocaleString()} patches in ${((t.embed + t.coreset + t.gather) / 1000).toFixed(1)} s`);
    });
  } catch (err) {
    fail(err);
  } finally {
    state.fitAbort = null;
    $('fit-cancel').hidden = true;
    $('fit-progress').hidden = true;
    $('fit-btn').disabled = !state.fitFiles.length;
  }
}

// ---------------------------------------------------------------- bank ----

function onBankActive(source) {
  const { header, features } = state.pc.bank;
  $('bank-card').hidden = false;
  $('threshold-card').hidden = false;
  $('bank-source').textContent = source;
  $('bank-name').value = header.name || 'bank';
  $('bank-count').textContent = header.count.toLocaleString();
  $('bank-size').textContent = mb(features.byteLength);
  $('bank-images').textContent = header.images ?? header.coreset?.images ?? '–';
  setThreshold(header.threshold ?? null, header.threshold_note || (header.threshold ? 'Stored with the bank' : null));
  disableModelUi(false);
  rescoreExisting();
}

function currentBankBuffer() {
  const { header, features } = state.pc.bank;
  header.name = $('bank-name').value.trim() || 'bank';
  header.threshold = state.threshold;
  return { buffer: encodeBank(features, header), header };
}

async function refreshSaved() {
  const list = $('saved-list');
  let banks = [];
  try { banks = await listBanks(); } catch (err) { console.warn(err); }
  list.innerHTML = '';
  if (!banks.length) {
    list.innerHTML = '<li class="empty">No saved banks yet.</li>';
    return;
  }
  for (const b of banks) {
    const li = document.createElement('li');
    const meta = document.createElement('div');
    meta.className = 'meta';
    const title = document.createElement('strong');
    title.textContent = b.name;
    const sub = document.createElement('span');
    sub.textContent = `${b.count.toLocaleString()} patches · ${mb(b.bytes)}${b.created ? ` · ${new Date(b.created).toLocaleDateString()}` : ''}`;
    meta.append(title, sub);
    const load = Object.assign(document.createElement('button'), { className: 'btn small', textContent: 'Load' });
    const del = Object.assign(document.createElement('button'), { className: 'btn small ghost', textContent: 'Delete' });
    load.onclick = () => exclusive(async () => {
      activateBuffer(await loadBank(b.id), 'Saved bank');
      state.bankId = b.id;
    }).catch(fail);
    del.onclick = async () => {
      if (!confirm(`Delete “${b.name}” from this browser?`)) return;
      await deleteBank(b.id).catch(fail);
      if (state.bankId === b.id) state.bankId = null;
      refreshSaved();
    };
    li.append(meta, load, del);
    list.append(li);
  }
}

function activateBuffer(buffer, source) {
  const bank = decodeBank(buffer);
  state.pc.setBank(bank);
  onBankActive(source);
}

// ----------------------------------------------------------- threshold ----

function setThreshold(value, note) {
  state.threshold = value;
  const slider = $('threshold');
  const maxSeen = Math.max(0, ...state.results.map((r) => r.score || 0));
  slider.max = Math.max(10, (value || 0) * 2.5, maxSeen * 1.2).toFixed(2);
  if (value !== null) {
    slider.value = value;
    $('threshold-num').value = value.toFixed(3);
  } else {
    $('threshold-num').value = '';
  }
  $('threshold-note').textContent = value === null
    ? 'Not calibrated. Score a few known-good images to set it.'
    : note || 'Set manually.';
  rerenderAll();
}

// ------------------------------------------------------------- inspect ----

function makeCard(file) {
  const btn = document.createElement('button');
  btn.className = 'result pending';
  btn.innerHTML = '<canvas width="224" height="224"></canvas><div class="info"><span class="name"></span>'
    + '<div class="row"><span class="score">…</span><span class="badge">queued</span></div></div>';
  btn.querySelector('.name').textContent = fileName(file);
  btn.title = fileName(file);
  return btn;
}

function verdict(score) {
  if (state.threshold === null || !Number.isFinite(score)) return null;
  return score >= state.threshold;
}

function renderCard(entry) {
  const { el, result } = entry;
  if (!result) return;
  el.classList.remove('pending');
  drawResult(el.querySelector('canvas'), result);
  el.querySelector('.score').textContent = fmt(result.score, 3);
  const badge = el.querySelector('.badge');
  const v = verdict(result.score);
  badge.className = `badge${v === null ? '' : v ? ' bad' : ' ok'}`;
  badge.textContent = v === null ? 'scored' : v ? 'anomaly' : 'ok';
}

function updateSummary() {
  const done = state.results.filter((r) => r.result);
  $('summary').hidden = !done.length;
  $('legend').hidden = !done.length;
  $('results-clear').disabled = !state.results.length;
  if (!done.length) return;
  const anomalies = done.filter((r) => verdict(r.result.score) === true).length;
  const avgMs = done.reduce((a, r) => a + r.result.timings.total, 0) / done.length;
  const parts = [`<span><b>${done.length}</b> inspected</span>`];
  if (state.threshold !== null) {
    parts.push(`<span class="bad"><b>${anomalies}</b> anomalous</span>`, `<span class="ok"><b>${done.length - anomalies}</b> ok</span>`);
  }
  parts.push(`<span><b>${avgMs.toFixed(0)}</b> ms / image</span>`);
  $('summary').innerHTML = parts.join('');
  const sample = done[done.length - 1].result;
  const { lo, hi } = mapRange(sample);
  const perImage = $('scale-mode').value === 'image' || !state.threshold;
  $('legend-lo').textContent = perImage ? 'low' : fmt(lo);
  $('legend-hi').textContent = perImage ? 'high' : fmt(hi);
}

function rerenderAll() {
  for (const r of state.results) renderCard(r);
  updateSummary();
}

function sortResults() {
  const mode = $('sort').value;
  const list = [...state.results];
  if (mode === 'score') list.sort((a, b) => (b.result?.score ?? -1) - (a.result?.score ?? -1));
  if (mode === 'name') list.sort((a, b) => fileName(a.file).localeCompare(fileName(b.file), undefined, { numeric: true }));
  $('results').append(...list.map((r) => r.el));
}

async function inspect(files) {
  const entries = files.map((file) => {
    const entry = { file, el: makeCard(file), result: null };
    entry.el.addEventListener('click', () => entry.result && openDetail(entry));
    return entry;
  });
  state.results.push(...entries);
  $('results').append(...entries.map((e) => e.el));
  $('results-clear').disabled = false;
  await scoreEntries(entries);
}

async function scoreEntries(entries) {
  await exclusive(async () => {
    for (const entry of entries) {
      if (!state.results.includes(entry)) continue; // cleared meanwhile
      entry.el.querySelector('.badge').textContent = 'scoring';
      try {
        entry.result = await state.pc.predict(entry.file);
        renderCard(entry);
      } catch (err) {
        entry.el.classList.remove('pending');
        entry.el.querySelector('.badge').textContent = 'error';
        entry.el.querySelector('.score').textContent = '';
        fail(err);
      }
      updateSummary();
    }
  });
  if ($('sort').value !== 'added') sortResults();
  const maxSeen = Math.max(...state.results.map((r) => r.result?.score || 0));
  if (maxSeen * 1.2 > Number($('threshold').max)) $('threshold').max = (maxSeen * 1.2).toFixed(2);
}

function rescoreExisting() {
  const entries = state.results.filter((r) => r.result || r.el.classList.contains('pending'));
  if (!entries.length) return;
  for (const e of entries) { e.result = null; e.el.classList.add('pending'); }
  scoreEntries(entries).catch(fail);
}

function openDetail(entry) {
  const { result, file } = entry;
  $('detail-name').textContent = fileName(file);
  const v = verdict(result.score);
  $('detail-meta').textContent = `Anomaly score ${fmt(result.score, 4)}`
    + (state.threshold !== null ? ` · threshold ${fmt(state.threshold, 3)} → ${v ? 'ANOMALY' : 'OK'}` : '');
  drawResult($('detail-image'), result, false);
  drawResult($('detail-overlay'), result, true);
  const t = result.timings;
  $('detail-timing').textContent = `preprocess ${t.preprocess.toFixed(0)} ms · features ${t.embed.toFixed(0)} ms · `
    + `nearest neighbour ${t.knn.toFixed(0)} ms · map ${t.post.toFixed(0)} ms`;
  $('detail-download').onclick = () => $('detail-overlay').toBlob((blob) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${fileName(file).replace(/[\\/]/g, '_').replace(/\.[^.]+$/, '')}_anomaly.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
  $('detail').showModal();
}

// ------------------------------------------------------------ evaluate ----

function updateEvalCounts() {
  $('eval-good-count').textContent = state.evalGood.length ? `${state.evalGood.length} images` : 'none';
  $('eval-bad-count').textContent = state.evalBad.length ? `${state.evalBad.length} images` : 'none';
  disableModelUi(!state.pc);
}

function bestF1(scores, labels) {
  const candidates = [...new Set(scores)].sort((a, b) => a - b);
  let best = { f1: -1, threshold: null };
  for (const t of candidates) {
    let tp = 0, fp = 0, fn = 0;
    scores.forEach((s, i) => {
      if (s >= t) labels[i] ? tp++ : fp++;
      else if (labels[i]) fn++;
    });
    const f1 = (2 * tp) / (2 * tp + fp + fn);
    if (f1 > best.f1) best = { f1, threshold: t };
  }
  return best;
}

function drawHistogram(scores, labels, threshold) {
  const svg = $('eval-hist');
  const W = svg.clientWidth || 600, H = 180, pad = 22, bins = 30;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const lo = Math.min(...scores), hi = Math.max(...scores);
  const span = hi - lo || 1;
  const counts = [new Array(bins).fill(0), new Array(bins).fill(0)];
  scores.forEach((s, i) => counts[labels[i]][Math.min(bins - 1, Math.floor(((s - lo) / span) * bins))]++);
  const peak = Math.max(1, ...counts[0], ...counts[1]);
  const bw = (W - 2 * pad) / bins;
  const x = (v) => pad + ((v - lo) / span) * (W - 2 * pad);
  let out = '';
  counts.forEach((c, label) => c.forEach((n, b) => {
    if (!n) return;
    const h = (n / peak) * (H - 2 * pad);
    out += `<rect class="${label ? 'bad' : 'good'}" x="${pad + b * bw + 1}" y="${H - pad - h}" width="${bw - 2}" height="${h}"/>`;
  }));
  out += `<line class="axis" x1="${pad}" x2="${W - pad}" y1="${H - pad}" y2="${H - pad}"/>`;
  if (threshold !== null) out += `<line class="thr" x1="${x(threshold)}" x2="${x(threshold)}" y1="${pad - 8}" y2="${H - pad}"/>`;
  out += `<text x="${pad}" y="${H - 6}">${fmt(lo)}</text><text x="${W - pad}" y="${H - 6}" text-anchor="end">${fmt(hi)}</text>`;
  out += `<text x="${pad}" y="12">■ good</text><text x="${pad + 64}" y="12" style="fill:var(--bad)">■ defective</text>`;
  svg.innerHTML = out;
}

async function runEval() {
  const items = [...state.evalGood.map((f) => [f, 0]), ...state.evalBad.map((f) => [f, 1])];
  $('eval-run').disabled = true;
  $('eval-progress').hidden = false;
  try {
    const scores = [];
    await exclusive(async () => {
      for (let i = 0; i < items.length; i++) {
        scores.push((await state.pc.predict(items[i][0])).score);
        $('eval-phase').textContent = `Scoring ${i + 1} / ${items.length}`;
        $('eval-bar').style.width = `${((i + 1) / items.length) * 100}%`;
      }
    });
    const labels = items.map((x) => x[1]);
    const auc = auroc(scores, labels);
    const best = bestF1(scores, labels);
    state.evalSuggestion = best.threshold;
    $('eval-out').hidden = false;
    $('eval-auroc').textContent = fmt(auc, 4);
    $('eval-f1').textContent = fmt(best.f1, 3);
    $('eval-thr').textContent = fmt(best.threshold, 3);
    drawHistogram(scores, labels, best.threshold);
  } catch (err) {
    fail(err);
  } finally {
    $('eval-progress').hidden = true;
    updateEvalCounts();
  }
}

// ---------------------------------------------------------------- wire ----

document.querySelectorAll('.segmented button').forEach((btn) => btn.addEventListener('click', () => {
  document.querySelectorAll('.segmented button').forEach((b) => b.setAttribute('aria-selected', String(b === btn)));
  document.querySelectorAll('.tab-panel').forEach((p) => { p.hidden = p.dataset.panel !== btn.dataset.tab; });
  if (btn.dataset.tab === 'load') refreshSaved();
}));

wireDrop($('fit-drop'), [$('fit-files'), $('fit-folder')], setFitFiles);
$('fit-clear').onclick = () => setFitFiles([]);
$('coreset').oninput = updateFitNote;
$('holdout').onchange = updateFitNote;
$('fit-btn').onclick = () => runFit();
$('fit-cancel').onclick = () => state.fitAbort?.abort();

$('bank-file').onchange = async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  try {
    const buffer = await file.arrayBuffer();
    await exclusive(() => activateBuffer(buffer, 'From file'));
    state.bankId = null;
  } catch (err) {
    fail(err);
  }
};
$('bank-save').onclick = async () => {
  try {
    const { buffer, header } = currentBankBuffer();
    state.bankId = await saveBank(buffer, { ...header, id: state.bankId || undefined });
    toast(`Saved “${header.name}” in this browser`);
    refreshSaved();
  } catch (err) {
    fail(err);
  }
};
$('bank-download').onclick = () => {
  const { buffer, header } = currentBankBuffer();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([buffer], { type: 'application/octet-stream' }));
  a.download = `${header.name.replace(/[^\w.-]+/g, '_')}.pcbank`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};

$('threshold').oninput = (e) => setThreshold(Number(e.target.value), 'Set manually.');
$('threshold-num').onchange = (e) => {
  const v = Number(e.target.value);
  if (Number.isFinite(v) && v > 0) setThreshold(v, 'Set manually.');
};
$('calib-files').onchange = async (e) => {
  const files = [...e.target.files].filter(isImage);
  e.target.value = '';
  if (!files.length) return;
  try {
    const scores = [];
    await exclusive(async () => {
      for (const f of files) scores.push((await state.pc.predict(f)).score);
    });
    setThreshold(Math.max(...scores), `Max score of ${files.length} good image${files.length > 1 ? 's' : ''}`);
    toast(`Threshold set to ${fmt(state.threshold, 3)}`);
  } catch (err) {
    fail(err);
  }
};

wireDrop($('inspect-drop'), [$('inspect-files'), $('inspect-folder')], (files) => inspect(files).catch(fail));
$('opacity').oninput = rerenderAll;
$('scale-mode').onchange = rerenderAll;
$('sort').onchange = sortResults;
$('results-clear').onclick = () => {
  state.results = [];
  $('results').innerHTML = '';
  updateSummary();
};

wireDrop($('eval-good'), [$('eval-good-files'), $('eval-good-folder')], (f) => { state.evalGood.push(...f); updateEvalCounts(); });
wireDrop($('eval-bad'), [$('eval-bad-files'), $('eval-bad-folder')], (f) => { state.evalBad.push(...f); updateEvalCounts(); });
$('eval-run').onclick = runEval;
$('eval-reset').onclick = () => {
  state.evalGood = [];
  state.evalBad = [];
  state.evalSuggestion = null;
  $('eval-out').hidden = true;
  updateEvalCounts();
};
$('eval-apply').onclick = () => {
  if (state.evalSuggestion !== null) setThreshold(state.evalSuggestion, 'Best-F1 threshold from evaluation');
};

$('open-settings').onclick = () => {
  const s = state.settings;
  $('set-backend').value = s.backend;
  $('set-precision').value = s.precision;
  if (s.modelId) $('set-model').value = s.modelId;
  $('set-cache').value = s.cacheMB;
  $('set-seed').value = s.seed;
  $('settings').showModal();
};
$('settings').addEventListener('close', () => {
  if ($('settings').returnValue !== 'apply') return;
  const prev = state.settings;
  const next = {
    backend: $('set-backend').value,
    precision: $('set-precision').value,
    modelId: $('set-model').value,
    cacheMB: Math.max(0, Number($('set-cache').value) || 0),
    seed: Number($('set-seed').value) || 0,
  };
  state.settings = next;
  storeSettings();
  if (next.backend !== prev.backend || next.precision !== prev.precision || next.modelId !== prev.modelId) {
    exclusive(initEngine).catch(fail);
  }
});

window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => e.preventDefault());

exclusive(initEngine);
