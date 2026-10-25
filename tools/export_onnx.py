"""Export the PatchCore feature extractor (and two helper graphs) to ONNX.

Run once; the browser app only needs the resulting files in ../models/.

    python tools/export_onnx.py                      # 224px, patchsize 3 (repo default)
    python tools/export_onnx.py --imagesize 320 --resize 366

Outputs
  models/embed_<tag>.onnx       image [B,3,S,S] -> patch features [B, P, 1024]
  models/embed_<tag>.fp16.onnx  same, fp16 weights (fp32 I/O) for WebGPU
  models/knn.onnx               query [Q,D], bank [M,D] -> min squared L2 [Q]
  models/matmul.onnx            a [N,K], b [K,J] -> a @ b (coreset projection)
  models/manifest.json          config the app reads at start-up

The embed graph reproduces patchcore.PatchCore._embed for backbone
"wideresnet50" with layers (layer2, layer3): PatchMaker unfold, bilinear
alignment of the deeper layer, MeanMapper preprocessing and the final
Aggregator, so the browser never re-implements those ops.
"""
import argparse
import hashlib
import json
import math
import os

import torch
import torch.nn.functional as F
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "models")


class AdaptivePoolLast(torch.nn.Module):
    """F.adaptive_avg_pool1d over the last axis, written with ONNX-friendly ops.

    adaptive_avg_pool1d with non-integer bin ratios (e.g. 4608 -> 1024) does not
    export, so bins are grouped by width: each group is a stride-1 AveragePool
    followed by a Gather at the bin start indices (stored as buffers so they
    export as single initializers).
    """

    def __init__(self, length, out_dim):
        super().__init__()
        self.length, self.out_dim = length, out_dim
        self.widths = []
        if length % out_dim == 0:
            return
        starts = [(i * length) // out_dim for i in range(out_dim)]
        ends = [math.ceil((i + 1) * length / out_dim) for i in range(out_dim)]
        order = []
        for w in sorted({e - s for s, e in zip(starts, ends)}):
            bins = [i for i in range(out_dim) if ends[i] - starts[i] == w]
            self.register_buffer(f"starts_{w}", torch.tensor([starts[i] for i in bins]))
            self.widths.append(w)
            order.extend(bins)
        inverse = [0] * out_dim
        for pos, i in enumerate(order):
            inverse[i] = pos
        self.register_buffer("inverse", torch.tensor(inverse) if order != sorted(order) else None)

    def forward(self, x):
        lead = x.shape[:-1]
        if not self.widths:
            return x.reshape(*lead, self.out_dim, self.length // self.out_dim).mean(-1)
        flat = x.reshape(-1, 1, self.length)
        parts = [
            F.avg_pool1d(flat, kernel_size=w, stride=1).flatten(1).index_select(-1, getattr(self, f"starts_{w}"))
            for w in self.widths
        ]
        out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        if self.inverse is not None:
            out = out.index_select(-1, self.inverse)
        return out.reshape(*lead, self.out_dim)


class Unfold(torch.nn.Module):
    """F.unfold(kernel_size=k, padding=(k-1)//2) as Pad + k*k shifted slices.

    Output channel c*k*k + ki*k + kj holds input channel c shifted by (ki, kj),
    the same order as torch.nn.Unfold, and the result is bit-exact. The plain
    F.unfold export is a long Range/Gather/Concat chain that exceeds WebGPU's
    storage-buffer limit, and a one-hot grouped conv hits an ORT WebGPU bug with
    channel-multiplier convolutions.
    """

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        k, pad = self.k, (self.k - 1) // 2
        c, h, w = x.shape[1:]
        padded = F.pad(x, (pad, pad, pad, pad))
        shifts = [padded[:, :, ki:ki + h, kj:kj + w] for ki in range(k) for kj in range(k)]
        return torch.stack(shifts, dim=2).reshape(-1, c * k * k, h * w)


class PatchCoreEmbed(torch.nn.Module):
    def __init__(self, patchsize, pretrain_dim, target_dim):
        super().__init__()
        net = torchvision.models.wide_resnet50_2(
            weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        )
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.patchsize = patchsize
        self.pretrain_dim = pretrain_dim
        self.target_dim = target_dim
        k = patchsize
        self.unfold2 = self.unfold3 = Unfold(k)
        self.pool2 = AdaptivePoolLast(512 * k * k, pretrain_dim)
        self.pool3 = AdaptivePoolLast(1024 * k * k, pretrain_dim)
        self.aggregate = AdaptivePoolLast(2 * pretrain_dim, target_dim)

    @torch.no_grad()
    def replicate_bn_warmup(self, imagesize):
        """Mirror a side effect of patchcore-inspection's PatchCore.load().

        load() calls NetworkFeatureAggregator.feature_dimensions(), which runs
        one forward pass on torch.ones while the torchvision backbone is still
        in train mode. That nudges every BatchNorm running mean/var (momentum
        0.1) up to layer3, and all of the repo's features are computed with
        those stats. Without this the embeddings differ by ~10%.
        """
        self.train()
        self.layer3(self.layer2(self.stem(torch.ones(1, 3, imagesize, imagesize))))
        self.eval()

    def forward(self, images):
        f2 = self.layer2(self.stem(images))
        f3 = self.layer3(f2)
        ref_h, ref_w = f2.shape[-2:]
        mapped = []
        for f, unfold, pool in ((f2, self.unfold2, self.pool2), (f3, self.unfold3, self.pool3)):
            h, w = f.shape[-2:]
            p = unfold(f)  # [B, C*k*k, H*W], channel order (C, ki, kj) as in PatchMaker.patchify
            if (h, w) != (ref_h, ref_w):
                ck = p.shape[1]
                p = p.reshape(-1, ck, h, w)
                p = F.interpolate(p, size=(ref_h, ref_w), mode="bilinear", align_corners=False)
                p = p.reshape(-1, ck, ref_h * ref_w)
            mapped.append(pool(p.transpose(1, 2)))
        stacked = torch.stack(mapped, dim=2).reshape(-1, ref_h * ref_w, 2 * self.pretrain_dim)
        return self.aggregate(stacked)


class NearestDistance(torch.nn.Module):
    """Squared L2 to the nearest bank row, like faiss.IndexFlatL2 with k=1."""

    def forward(self, query, bank):
        q2 = (query * query).sum(1, keepdim=True)
        b2 = (bank * bank).sum(1).unsqueeze(0)
        d = q2 - 2.0 * (query @ bank.transpose(0, 1)) + b2
        return d.min(dim=1).values.clamp_min(0.0)


class MatMul(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


def export(module, args, path, input_names, output_names, dynamic_axes):
    torch.onnx.export(
        module, args, path,
        input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes, opset_version=17, dynamo=False,
    )
    print("wrote", os.path.relpath(path), f"{os.path.getsize(path) / 1e6:.1f} MB")


def file_entry(path):
    """Manifest record; the hash lets the browser invalidate its model cache."""
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    return {"path": os.path.basename(path), "bytes": os.path.getsize(path), "sha256": digest}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagesize", type=int, default=224)
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--patchsize", type=int, default=3)
    ap.add_argument("--pretrain-dim", type=int, default=1024)
    ap.add_argument("--target-dim", type=int, default=1024)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--no-bn-warmup", action="store_true",
                    help="use pristine ImageNet BatchNorm stats instead of matching the repo")
    args = ap.parse_args()

    os.makedirs(MODELS, exist_ok=True)
    tag = f"wrn50_l2l3_s{args.imagesize}_p{args.patchsize}"
    model = PatchCoreEmbed(args.patchsize, args.pretrain_dim, args.target_dim).eval()
    if not args.no_bn_warmup:
        model.replicate_bn_warmup(args.imagesize)
    else:
        tag += "_nowarmup"
    dummy = torch.randn(1, 3, args.imagesize, args.imagesize)
    with torch.no_grad():
        out = model(dummy)
    num_patches = out.shape[1]
    grid = int(round(math.sqrt(num_patches)))

    embed_path = os.path.join(MODELS, f"embed_{tag}.onnx")
    export(model, (dummy,), embed_path, ["image"], ["features"],
           {"image": {0: "batch"}, "features": {0: "batch"}})

    files = {"fp32": file_entry(embed_path)}
    if not args.no_fp16:
        import onnx
        from onnxconverter_common import float16

        fp16 = float16.convert_float_to_float16(onnx.load(embed_path), keep_io_types=True)
        fp16_path = embed_path.replace(".onnx", ".fp16.onnx")
        onnx.save(fp16, fp16_path)
        files["fp16"] = file_entry(fp16_path)
        print("wrote", os.path.relpath(fp16_path), f"{os.path.getsize(fp16_path) / 1e6:.1f} MB")

    d = args.target_dim
    export(NearestDistance(), (torch.randn(8, d), torch.randn(16, d)),
           os.path.join(MODELS, "knn.onnx"), ["query", "bank"], ["distance"],
           {"query": {0: "Q"}, "bank": {0: "M"}, "distance": {0: "Q"}})
    export(MatMul(), (torch.randn(8, d), torch.randn(d, 128)),
           os.path.join(MODELS, "matmul.onnx"), ["a", "b"], ["c"],
           {"a": {0: "N", 1: "K"}, "b": {0: "K", 1: "J"}, "c": {0: "N", 1: "J"}})

    manifest_path = os.path.join(MODELS, "manifest.json")
    manifest = {"models": []}
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    entry = {
        "id": tag,
        "backbone": "wideresnet50",
        "layers": ["layer2", "layer3"],
        "resize": args.resize,
        "imagesize": args.imagesize,
        "patchsize": args.patchsize,
        "grid": [grid, grid],
        "dim": d,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "files": files,
    }
    manifest["models"] = [m for m in manifest["models"] if m["id"] != tag] + [entry]
    manifest["knn"] = file_entry(os.path.join(MODELS, "knn.onnx"))
    manifest["matmul"] = file_entry(os.path.join(MODELS, "matmul.onnx"))
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("wrote", os.path.relpath(manifest_path))


if __name__ == "__main__":
    main()
