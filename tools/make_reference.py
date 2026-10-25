"""Run patchcore-inspection on one MVTec category and dump parity data for tests/parity.html.

    python tools/make_reference.py --patchcore-src ../patchcore-inspection/src \
        --mvtec data/mvtec --category bottle

Writes reference/<category>/:
  model/            PatchCore.save_to_path() output (exercise export_bank.py on it)
  bank.pcbank       exported memory bank
  reference.json    per-test-image scores, labels, AUROC, and which maps are stored
  maps.bin          float32 224x224 anomaly maps for a subset of test images
  prep.bin          float32 [3,S,S] preprocessed tensor of the first test image
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patchcore-src", required=True)
    ap.add_argument("--mvtec", required=True)
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--resize", type=int, default=256)
    ap.add_argument("--imagesize", type=int, default=224)
    ap.add_argument("--maps", type=int, default=12, help="number of anomaly maps to store")
    args = ap.parse_args()
    sys.path.insert(0, args.patchcore_src)
    import patchcore.common
    import patchcore.datasets.mvtec as mvtec
    import patchcore.metrics
    import patchcore.patchcore
    import patchcore.sampler
    import patchcore.utils
    from sklearn.metrics import roc_auc_score

    import export_bank
    import pcbank

    out_dir = os.path.join(ROOT, "reference", args.category)
    os.makedirs(out_dir, exist_ok=True)
    patchcore.utils.fix_seeds(0)
    device = torch.device("cpu")

    def loader(split):
        ds = mvtec.MVTecDataset(args.mvtec, args.category, resize=args.resize,
                                imagesize=args.imagesize, split=split)
        return torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=4)

    backbone = torchvision.models.wide_resnet50_2(
        weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    backbone.name = "wideresnet50"
    pc = patchcore.patchcore.PatchCore(device)
    pc.load(backbone=backbone, layers_to_extract_from=["layer2", "layer3"], device=device,
            input_shape=(3, args.imagesize, args.imagesize), pretrain_embed_dimension=1024,
            target_embed_dimension=1024, patchsize=3, anomaly_score_num_nn=1,
            featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(args.coreset, device),
            nn_method=patchcore.common.FaissNN(False, 4))
    pc.fit(loader(mvtec.DatasetSplit.TRAIN))

    model_dir = os.path.join(out_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    pc.save_to_path(model_dir)
    features, params = export_bank.load_saved_model(model_dir)
    pcbank.write(os.path.join(out_dir, "bank.pcbank"), features,
                 export_bank.header_from_params(params, args.resize, "patchcore-inspection",
                                                name=f"mvtec_{args.category}"))

    test = loader(mvtec.DatasetSplit.TEST)
    scores, masks, labels, masks_gt = pc.predict(test)
    scores = np.asarray(scores, dtype=np.float64)
    masks = np.asarray(masks, dtype=np.float32)
    labels = np.asarray(labels)
    image_auroc = roc_auc_score(labels, scores)
    pixel_auroc = patchcore.metrics.compute_pixelwise_retrieval_metrics(
        masks, np.asarray(masks_gt))["auroc"]

    paths = [os.path.relpath(t[2], ROOT) for t in test.dataset.data_to_iterate]
    good = [i for i in range(len(paths)) if labels[i] == 0]
    bad = [i for i in range(len(paths)) if labels[i] == 1]
    n_good = min(len(good), args.maps // 3)
    pick = sorted(good[:: max(1, len(good) // n_good)][:n_good]
                  + bad[:: max(1, len(bad) // (args.maps - n_good))][: args.maps - n_good])
    masks[pick].astype("<f4").tofile(os.path.join(out_dir, "maps.bin"))
    test.dataset[0]["image"].numpy().astype("<f4").tofile(os.path.join(out_dir, "prep.bin"))

    ref = {
        "category": args.category,
        "bank": "bank.pcbank",
        "imagesize": args.imagesize,
        "resize": args.resize,
        "coreset": args.coreset,
        "bank_size": int(features.shape[0]),
        "image_auroc": float(image_auroc),
        "pixel_auroc": float(pixel_auroc),
        "images": [{"path": p, "label": int(l), "score": float(s)}
                   for p, l, s in zip(paths, labels, scores)],
        "maps": pick,
        "prep_image": paths[0],
        "train_images": [os.path.relpath(t[2], ROOT)
                         for t in loader(mvtec.DatasetSplit.TRAIN).dataset.data_to_iterate],
    }
    with open(os.path.join(out_dir, "reference.json"), "w") as fh:
        json.dump(ref, fh, indent=1)
    print(f"{args.category}: bank {features.shape}, image AUROC {image_auroc:.4f}, "
          f"pixel AUROC {pixel_auroc:.4f} -> {os.path.relpath(out_dir)}")


if __name__ == "__main__":
    main()
