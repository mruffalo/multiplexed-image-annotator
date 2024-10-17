from argparse import ArgumentParser
from pathlib import Path

import torch
from .utils import gui_run

cuda_available = torch.cuda.is_available()
print(f"{cuda_available=}")


def main(
    marker_list_path: Path,
    image_path: Path,
    mask_path: Path,
    results_dir: Path,
):
    results_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if cuda_available else "cpu")
    batch_id = ""
    strict = False
    normalization = True
    blur = 0.5
    confidence = 0.25
    batch_size = 1
    amax = 1
    cell_size = 30

    gui_run(
        marker_list_path=marker_list_path,
        image_path=image_path,
        mask_path=mask_path,
        device=device,
        main_dir=results_dir,
        batch_id=batch_id,
        bs=batch_size,
        strict=strict,
        infer=False,
        normalization=normalization,
        blur=blur,
        confidence=confidence,
        amax=amax,
        cell_size=cell_size,
        cell_type_confidence=None,
    )


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("marker_list_path", type=Path)
    p.add_argument("image_path", type=Path)
    p.add_argument("mask_path", type=Path)
    p.add_argument(
        "results_dir",
        type=Path,
        default=Path("results"),
        nargs="?",
    )
    args = p.parse_args()

    main(
        marker_list_path=args.marker_list_path,
        image_path=args.image_path,
        mask_path=args.mask_path,
        results_dir=args.results_dir,
    )
