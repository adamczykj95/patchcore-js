// Image preprocessing identical to the torchvision pipeline used by
// patchcore-inspection: Resize(resize) -> CenterCrop(imagesize) -> ToTensor -> Normalize.
// Resize is a port of Pillow's bilinear resampler (src/libImaging/Resample.c),
// including its 22-bit fixed-point coefficients and uint8 rounding between the
// horizontal and vertical passes, so pixels match PIL exactly.

const PRECISION_BITS = 32 - 8 - 2;
const ONE = 2 ** PRECISION_BITS;

export const IMAGE_EXTENSIONS = /\.(png|jpe?g|bmp|webp|gif|tiff?)$/i;

/** Decode a File/Blob/URL into packed 8-bit RGB without colour management. */
export async function decodeImage(source) {
  const blob = typeof source === 'string' ? await (await fetch(source)).blob() : source;
  const bitmap = await createImageBitmap(blob, {
    colorSpaceConversion: 'none',
    premultiplyAlpha: 'none',
  });
  const { width, height } = bitmap;
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = ctx.getImageData(0, 0, width, height).data;
  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i];
    rgb[j + 1] = rgba[i + 1];
    rgb[j + 2] = rgba[i + 2];
  }
  return { width, height, rgb };
}

/** torchvision Resize(int): shorter side -> size, longer side truncated. */
export function resizedSize(width, height, size) {
  if (width <= height) return [size, Math.trunc((size * height) / width)];
  return [Math.trunc((size * width) / height), size];
}

function precomputeCoeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = filterscale; // bilinear support = 1
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  const tmp = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const t = Math.abs((x + xmin - center + 0.5) / filterscale);
      const w = t < 1 ? 1 - t : 0;
      tmp[x] = w;
      ww += w;
    }
    for (let x = 0; x < xmax; x++) {
      const k = ww !== 0 ? tmp[x] / ww : tmp[x];
      kk[xx * ksize + x] = k < 0 ? Math.trunc(-0.5 + k * ONE) : Math.trunc(0.5 + k * ONE);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { bounds, kk, ksize };
}

function clip8(v) {
  if (v >= ONE * 256) return 255;
  if (v <= 0) return 0;
  return Math.floor(v / ONE);
}

/** Pillow Image.resize((outW, outH), Image.BILINEAR) on packed RGB. */
export function pilResizeBilinear(rgb, width, height, outW, outH) {
  let src = rgb;
  let w = width;
  if (outW !== width) {
    const { bounds, kk, ksize } = precomputeCoeffs(width, outW);
    const dst = new Uint8Array(outW * height * 3);
    for (let y = 0; y < height; y++) {
      const row = y * width * 3;
      for (let xx = 0; xx < outW; xx++) {
        const xmin = bounds[xx * 2];
        const xmax = bounds[xx * 2 + 1];
        const k = xx * ksize;
        let s0 = ONE / 2, s1 = ONE / 2, s2 = ONE / 2;
        for (let x = 0; x < xmax; x++) {
          const p = row + (x + xmin) * 3;
          const c = kk[k + x];
          s0 += src[p] * c;
          s1 += src[p + 1] * c;
          s2 += src[p + 2] * c;
        }
        const o = (y * outW + xx) * 3;
        dst[o] = clip8(s0);
        dst[o + 1] = clip8(s1);
        dst[o + 2] = clip8(s2);
      }
    }
    src = dst;
    w = outW;
  }
  if (outH !== height) {
    const { bounds, kk, ksize } = precomputeCoeffs(height, outH);
    const dst = new Uint8Array(w * outH * 3);
    for (let yy = 0; yy < outH; yy++) {
      const ymin = bounds[yy * 2];
      const ymax = bounds[yy * 2 + 1];
      const k = yy * ksize;
      for (let x = 0; x < w; x++) {
        let s0 = ONE / 2, s1 = ONE / 2, s2 = ONE / 2;
        for (let y = 0; y < ymax; y++) {
          const p = ((y + ymin) * w + x) * 3;
          const c = kk[k + y];
          s0 += src[p] * c;
          s1 += src[p + 1] * c;
          s2 += src[p + 2] * c;
        }
        const o = (yy * w + x) * 3;
        dst[o] = clip8(s0);
        dst[o + 1] = clip8(s1);
        dst[o + 2] = clip8(s2);
      }
    }
    src = dst;
  }
  return src === rgb ? rgb.slice() : src;
}

// Python's round(): halves go to the even neighbour.
function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** torchvision CenterCrop(size) on packed RGB (image must be >= size). */
export function centerCrop(rgb, width, height, size) {
  const top = roundHalfEven((height - size) / 2);
  const left = roundHalfEven((width - size) / 2);
  const out = new Uint8Array(size * size * 3);
  for (let y = 0; y < size; y++) {
    const s = ((y + top) * width + left) * 3;
    out.set(rgb.subarray(s, s + size * 3), y * size * 3);
  }
  return out;
}

/** ToTensor + Normalize -> Float32Array laid out [3, size, size]. */
export function normalize(rgb, size, mean, std) {
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  for (let c = 0; c < 3; c++) {
    const m = Math.fround(mean[c]);
    const s = Math.fround(std[c]);
    for (let i = 0; i < plane; i++) {
      out[c * plane + i] = Math.fround(Math.fround(rgb[i * 3 + c] / 255) - m) / s;
    }
  }
  return out;
}

/**
 * Full pipeline. Returns the model input and the cropped RGB (what the model
 * sees), which the UI uses as the background for the anomaly overlay.
 */
export async function preprocess(source, cfg) {
  const img = await decodeImage(source);
  const [rw, rh] = resizedSize(img.width, img.height, cfg.resize);
  const resized = pilResizeBilinear(img.rgb, img.width, img.height, rw, rh);
  const crop = centerCrop(resized, rw, rh, cfg.imagesize);
  return {
    tensor: normalize(crop, cfg.imagesize, cfg.mean, cfg.std),
    rgb: crop,
    size: cfg.imagesize,
    original: { width: img.width, height: img.height },
  };
}
