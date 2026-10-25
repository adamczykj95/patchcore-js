// Turning nearest-neighbour patch distances into PatchCore outputs, matching
// patchcore.common.RescaleSegmentor and PatchMaker.score.

/** torch F.interpolate(mode="bilinear", align_corners=False) for one 2-D map. */
export function upsampleBilinear(src, inH, inW, outH, outW) {
  const out = new Float32Array(outH * outW);
  const sy = inH / outH;
  const sx = inW / outW;
  const x0s = new Int32Array(outW);
  const x1s = new Int32Array(outW);
  const lxs = new Float32Array(outW);
  for (let x = 0; x < outW; x++) {
    const fx = Math.max((x + 0.5) * sx - 0.5, 0);
    x0s[x] = Math.floor(fx);
    x1s[x] = Math.min(x0s[x] + 1, inW - 1);
    lxs[x] = fx - x0s[x];
  }
  for (let y = 0; y < outH; y++) {
    const fy = Math.max((y + 0.5) * sy - 0.5, 0);
    const y0 = Math.floor(fy);
    const y1 = Math.min(y0 + 1, inH - 1);
    const ly = fy - y0;
    for (let x = 0; x < outW; x++) {
      const lx = lxs[x];
      const a = src[y0 * inW + x0s[x]], b = src[y0 * inW + x1s[x]];
      const c = src[y1 * inW + x0s[x]], d = src[y1 * inW + x1s[x]];
      out[y * outW + x] = (1 - ly) * ((1 - lx) * a + lx * b) + ly * ((1 - lx) * c + lx * d);
    }
  }
  return out;
}

// scipy.ndimage "reflect" boundary (d c b a | a b c d | d c b a).
function reflectIndex(i, n) {
  const period = 2 * n;
  i %= period;
  if (i < 0) i += period;
  return i < n ? i : period - 1 - i;
}

/** scipy.ndimage.gaussian_filter(img, sigma, truncate=4.0, mode="reflect") on float32. */
export function gaussianFilter(src, h, w, sigma = 4, truncate = 4) {
  const radius = Math.trunc(truncate * sigma + 0.5);
  const kernel = new Float64Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    kernel[i + radius] = Math.exp((-0.5 / (sigma * sigma)) * i * i);
    sum += kernel[i + radius];
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;

  // axis 0 (rows) first, then axis 1, each pass stored as float32 like scipy.
  const tmp = new Float32Array(h * w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) acc += kernel[k + radius] * src[reflectIndex(y + k, h) * w + x];
      tmp[y * w + x] = acc;
    }
  }
  const out = new Float32Array(h * w);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) acc += kernel[k + radius] * tmp[row + reflectIndex(x + k, w)];
      out[row + x] = acc;
    }
  }
  return out;
}

/**
 * patchScores: Float32Array of grid[0]*grid[1] nearest-neighbour distances.
 * Returns the image score (max patch score) and the smoothed full-size map.
 */
export function buildResult(patchScores, grid, imagesize, sigma = 4) {
  let score = -Infinity;
  for (const v of patchScores) if (v > score) score = v;
  const up = upsampleBilinear(patchScores, grid[0], grid[1], imagesize, imagesize);
  const map = gaussianFilter(up, imagesize, imagesize, sigma);
  return { score, patchScores, map, size: imagesize };
}

/** ROC AUC with average ranks for ties (equals sklearn.metrics.roc_auc_score). */
export function auroc(scores, labels) {
  const idx = scores.map((s, i) => i).sort((a, b) => scores[a] - scores[b]);
  const ranks = new Float64Array(scores.length);
  for (let i = 0; i < idx.length; ) {
    let j = i;
    while (j + 1 < idx.length && scores[idx[j + 1]] === scores[idx[i]]) j++;
    const r = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[idx[k]] = r;
    i = j + 1;
  }
  let pos = 0, rankSum = 0;
  labels.forEach((l, i) => {
    if (l) { pos++; rankSum += ranks[i]; }
  });
  const neg = labels.length - pos;
  if (!pos || !neg) return NaN;
  return (rankSum - (pos * (pos + 1)) / 2) / (pos * neg);
}

/** Pearson correlation, used by the parity test. */
export function correlation(a, b) {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    sab += da * db; saa += da * da; sbb += db * db;
  }
  return sab / Math.sqrt(saa * sbb);
}
