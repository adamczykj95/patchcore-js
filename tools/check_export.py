"""Compare the exported embed graph against patchcore-inspection's PatchCore._embed.

    python tools/check_export.py --patchcore-src /path/to/patchcore-inspection/src
"""
import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patchcore-src", required=True)
    ap.add_argument("--model-id", default="wrn50_l2l3_s224_p3")
    args = ap.parse_args()
    sys.path.insert(0, args.patchcore_src)
    import patchcore.patchcore

    with open(os.path.join(MODELS, "manifest.json")) as fh:
        cfg = next(m for m in json.load(fh)["models"] if m["id"] == args.model_id)

    backbone = torchvision.models.wide_resnet50_2(
        weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
    )
    backbone.name = "wideresnet50"
    pc = patchcore.patchcore.PatchCore(torch.device("cpu"))
    size = cfg["imagesize"]
    pc.load(backbone=backbone, layers_to_extract_from=cfg["layers"], device=torch.device("cpu"),
            input_shape=(3, size, size), pretrain_embed_dimension=cfg["dim"],
            target_embed_dimension=cfg["dim"], patchsize=cfg["patchsize"])

    torch.manual_seed(0)
    images = torch.randn(2, 3, size, size)
    ref = np.asarray(pc._embed(images)).reshape(2, -1, cfg["dim"])

    for kind, entry in cfg["files"].items():
        sess = ort.InferenceSession(os.path.join(MODELS, entry["path"]), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"image": images.numpy()})[0]
        err = np.abs(out - ref)
        rel = err.max() / np.abs(ref).max()
        print(f"{kind}: shape {out.shape} max|diff| {err.max():.2e} mean|diff| {err.mean():.2e} "
              f"rel {rel:.2e} (ref max {np.abs(ref).max():.2f})")


if __name__ == "__main__":
    main()
