"""Convert a model saved by patchcore-inspection into a .pcbank for the browser.

    python tools/export_bank.py results/.../models/mvtec_bottle  bottle.pcbank

The folder must contain <prepend>patchcore_params.pkl and
<prepend>nnscorer_search_index.faiss as written by PatchCore.save_to_path().
Requires `pip install faiss-cpu`. Only exact (IndexFlatL2) indices can be
exported; models trained with --faiss_approx store lossy PQ codes.
"""
import argparse
import datetime
import glob
import os
import pickle
import sys

import pcbank


def load_saved_model(folder, prepend=None):
    import faiss

    if prepend is None:
        params = sorted(glob.glob(os.path.join(folder, "*patchcore_params.pkl")))
        if not params:
            raise FileNotFoundError(f"no *patchcore_params.pkl in {folder}")
        if len(params) > 1:
            print(f"note: {len(params)} models (ensemble) found, exporting {os.path.basename(params[0])}",
                  file=sys.stderr)
        prepend = os.path.basename(params[0])[: -len("patchcore_params.pkl")]
    with open(os.path.join(folder, prepend + "patchcore_params.pkl"), "rb") as fh:
        params = pickle.load(fh)
    index = faiss.read_index(os.path.join(folder, prepend + "nnscorer_search_index.faiss"))
    if not isinstance(index, faiss.IndexFlat):
        raise ValueError(f"index type {type(index).__name__} is approximate; re-train without --faiss_approx")
    return index.reconstruct_n(0, index.ntotal), params


def header_from_params(params, resize, source, **extra):
    imagesize = int(params["input_shape"][-1])
    return {
        "model": {
            "backbone": params["backbone.name"],
            "layers": list(params["layers_to_extract_from"]),
            "imagesize": imagesize,
            "resize": resize or round(imagesize * 256 / 224),
            "patchsize": int(params["patchsize"]),
            "dim": int(params["target_embed_dimension"]),
        },
        "num_nn": int(params["anomaly_scorer_num_nn"]),
        "source": source,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--prepend", default=None, help="model file prefix, e.g. 'Ensemble-1-1_'")
    ap.add_argument("--resize", type=int, default=None, help="dataset --resize used in training")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    features, params = load_saved_model(args.model_dir, args.prepend)
    header = header_from_params(params, args.resize, "patchcore-inspection",
                                name=args.name or os.path.basename(os.path.normpath(args.model_dir)))
    m = header["model"]
    if m["backbone"] != "wideresnet50" or m["layers"] != ["layer2", "layer3"]:
        print(f"warning: backbone {m['backbone']} {m['layers']} has no exported ONNX graph; "
              "only wideresnet50 layer2+layer3 is supported", file=sys.stderr)
    elif (m["imagesize"], m["patchsize"]) != (224, 3):
        print(f"note: export a matching graph with: python tools/export_onnx.py "
              f"--imagesize {m['imagesize']} --resize {m['resize']} --patchsize {m['patchsize']}",
              file=sys.stderr)
    pcbank.write(args.out, features, header)
    print(f"wrote {args.out}: {features.shape[0]} x {features.shape[1]} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
