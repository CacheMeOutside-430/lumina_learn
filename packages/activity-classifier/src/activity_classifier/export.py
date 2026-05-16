from __future__ import annotations

import argparse
from pathlib import Path

import torch

from activity_classifier.model import ScreenActivityNet


def export_onnx(checkpoint: Path, output: Path, image_size: int = 224) -> Path:
    device = torch.device("cpu")
    model = ScreenActivityNet().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload.get("model_state_dict", payload))
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    torch.onnx.export(
        model,
        (dummy,),
        output,
        input_names=["screen"],
        output_names=["activity_logits", "education_logits"],
        dynamic_axes={
            "screen": {0: "batch"},
            "activity_logits": {0: "batch"},
            "education_logits": {0: "batch"},
        },
        opset_version=17,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the activity classifier to ONNX.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    print({"onnx": str(export_onnx(args.checkpoint, args.output, args.image_size))}, flush=True)


if __name__ == "__main__":
    main()
