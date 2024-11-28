import json
from argparse import ArgumentParser
from pathlib import Path
from pprint import pprint
from typing import Optional

import torch

from .gui_api import headless_run

cuda_available = torch.cuda.is_available()
print(f"{cuda_available=}")


def main(
    marker_list_path: Path,
    image_path: Path,
    mask_path: Path,
    hyperparameters_path: Optional[Path] = None,
):
    results_dir = Path()

    device = torch.device("cuda" if cuda_available else "cpu")

    hyperparameters = {}
    if hyperparameters_path is not None:
        print("Loading hyperparameters from", hyperparameters_path)
        with open(hyperparameters_path) as f:
            hyperparameters = json.load(f)

    batch_id = "headless"
    strict = hyperparameters.get("strict", False)
    infer = hyperparameters.get("infer", True)
    normalization = hyperparameters.get("normalize", True)
    blur = hyperparameters.get("blur", 0.3)
    confidence = hyperparameters.get("confidence", 0.3)
    batch_size = 1
    amax = hyperparameters.get("upper_limit", 99.8)
    cell_size = hyperparameters.get("cell_size", 30)
    cell_type_confidence = hyperparameters.get("cell_type_confidence")

    kwargs = {
        "marker_list_path": marker_list_path,
        "image_path": image_path,
        "mask_path": mask_path,
        "device": device,
        "main_dir": results_dir,
        "batch_id": batch_id,
        "bs": batch_size,
        "strict": strict,
        "infer": infer,
        "normalization": normalization,
        "blur": blur,
        "confidence": confidence,
        "amax": amax,
        "cell_size": cell_size,
        "cell_type_confidence": cell_type_confidence,
    }

    print("Starting headless run with parameters:")
    pprint(kwargs)

    headless_run(**kwargs)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("marker_list_path", type=Path)
    p.add_argument("image_path", type=Path)
    p.add_argument("mask_path", type=Path)
    p.add_argument("hyperparameters_path", type=Path, nargs="?")
    args = p.parse_args()

    main(
        marker_list_path=args.marker_list_path,
        image_path=args.image_path,
        mask_path=args.mask_path,
        hyperparameters_path=args.hyperparameters_path,
    )
