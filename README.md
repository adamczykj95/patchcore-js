# PatchCore Inspector

[PatchCore](https://github.com/amazon-science/patchcore-inspection) image anomaly detection that runs entirely in the browser using plain HTML, CSS and JavaScript. It has no build step and no server-side inference, and images never leave the device.

**Live demo:** https://adamczykj95.github.io/patchcore-js/ (Chrome or Edge for WebGPU; the first visit downloads the 100 MB model, which is then cached).

- **Fit** a memory bank from defect-free images in the browser, or **load** one converted from a model trained with patchcore-inspection.
- **Inspect** images and get an anomaly score plus a heatmap overlay. You can calibrate a threshold, evaluate AUROC on labelled images, and save banks to IndexedDB or download them as `.pcbank` files.
- **Backends:** WebGPU (about 60 ms per image on an RTX 3060) or WebAssembly on the CPU (about 200 ms per image with 8 threads), via [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) 1.27.0 loaded from jsDelivr.

## Run

```sh
python serve.py          # http://localhost:8000
```

Any static server works. `serve.py` adds the COOP/COEP headers that make the page cross-origin isolated. Isolation enables multithreaded WASM and the multi-worker CPU coreset; without it, both fall back to a single thread. ES modules need `http://`, so opening `index.html` from disk won't work.

The ONNX models in `models/` are generated once (see below).

## How it maps to PatchCore

| PatchCore (Python) | Here |
|---|---|
| WideResNet-50 `layer2` + `layer3`, `PatchMaker` 3×3 unfold, `MeanMapper`/`Aggregator` to 1024-d | One ONNX graph: `models/embed_*.onnx`, from `tools/export_onnx.py` |
| torchvision `Resize(256)` → `CenterCrop(224)` → `Normalize` | `js/preprocess.js`, a port of Pillow's fixed-point bilinear resampler (bit-exact) |
| `ApproximateGreedyCoresetSampler` (128-d random projection, 10 start points) | `js/coreset.js`: a WebGPU compute-shader loop, or CPU Web Workers sharing memory |
| `faiss.IndexFlatL2`, k = 1 | `js/knn.js` + `models/knn.onnx`; the bank stays on the GPU between queries |
| `RescaleSegmentor` (bilinear upsample + Gaussian σ = 4), image score = max patch score | `js/scoring.js`, matching `F.interpolate` and `scipy.ndimage.gaussian_filter` |

The export also reproduces one quirk of the reference. `PatchCore.load()` runs one forward pass while the torchvision backbone is still in train mode, which shifts the BatchNorm running statistics. Every score the repo produces depends on that shift, and without it the features differ by about 10% (`--no-bn-warmup` disables it).

## Python tools (one-time, optional)

```sh
python -m venv .venv && .venv/bin/pip install -r tools/requirements.txt
.venv/bin/python tools/export_onnx.py                  # models/ for 224 px, patch size 3
.venv/bin/python tools/export_onnx.py --imagesize 320 --resize 366   # extra variant
.venv/bin/python tools/check_export.py --patchcore-src ../patchcore-inspection/src
```

To convert a model saved by patchcore-inspection (`--save_patchcore_model`) to a `.pcbank`:

```sh
.venv/bin/python tools/export_bank.py results/.../models/mvtec_bottle bottle.pcbank
```

Only exact faiss indexes can be converted (models trained without `--faiss_approx`). Scoring uses k = 1, the default `--anomaly_scorer_num_nn`. The `.pcbank` format is a JSON header followed by float32 rows; it is documented in `js/bank.js` and `tools/pcbank.py`.

## Parity test

`tools/make_reference.py` runs patchcore-inspection on an MVTec AD category and writes reference scores, maps and a bank to `reference/<category>/`. `tests/parity.html` checks the browser against them:

```sh
.venv/bin/python tools/make_reference.py --patchcore-src ../patchcore-inspection/src --mvtec data/mvtec --category bottle
# then open http://localhost:8000/tests/parity.html?fit=1   (&backend=wasm, &coreset=0.02, &limit=N)
```

Results on MVTec `bottle` (83 test images), Chrome 154 on Linux with an RTX 3060:

| Check | WebGPU | WASM |
|---|---|---|
| Preprocessing vs torchvision/PIL, max abs diff | 0 | 0 |
| Image score vs Python, max relative error | 4.2e-5 | 3.3e-5 |
| Anomaly map correlation (min over 12 maps) | 1.00000 | 1.00000 |
| Image AUROC, Python bank | 1.0000 (Python: 1.0000) | 1.0000 |
| Image AUROC, bank fitted in browser | 1.0000 (10% coreset, about 27 s for 209 images) | 1.0000 (2% coreset, about 32 s) |

## Limitations

- **Backbone:** only WideResNet-50 with layer2+layer3 is exported. Other backbones need their own export.
- **Ensembles:** banks with k > 1 and ensembles are not supported.
- **CPU coreset speed:** without WebGPU, a 10% coreset over about 150k patches takes about 45 s with 12 workers, or about 5 minutes on one thread. Lower the coreset percentage on slow machines.
- **fp16 model:** it halves the download but needs WebGPU with `shader-f16`. Otherwise the app falls back to fp32.
- **Memory:** fitting holds every patch feature in memory until the coreset is chosen, about 3.2 MB per image. Above the cache limit in Settings, images are embedded a second time instead.
- **JPEG decoding:** browser decoders can differ from libjpeg by ±1 per pixel, so exact parity is guaranteed for PNG.
