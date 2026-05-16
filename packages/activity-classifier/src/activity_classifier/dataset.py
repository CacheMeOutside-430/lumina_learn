from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from activity_classifier.labels import ACTIVITY_TO_INDEX, EDUCATION_TO_INDEX


@dataclass(frozen=True)
class ManifestRow:
    image_path: Path
    activity_index: int
    education_index: int


def default_train_transform(image_size: int = 224) -> Callable[[Image.Image], torch.Tensor]:
    return cast(
        Callable[[Image.Image], torch.Tensor],
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.04),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
    )


def default_eval_transform(image_size: int = 224) -> Callable[[Image.Image], torch.Tensor]:
    return cast(
        Callable[[Image.Image], torch.Tensor],
        transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
    )


class ScreenshotManifestDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        manifest_path: str | Path,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.transform = transform or default_train_transform()
        self.rows = self._load_manifest(self.manifest_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with Image.open(row.image_path) as raw_image:
            image = raw_image.convert("RGB")
        tensor = self.transform(image)
        return (
            tensor,
            torch.tensor(row.activity_index, dtype=torch.long),
            torch.tensor(row.education_index, dtype=torch.long),
        )

    @staticmethod
    def _load_manifest(path: Path) -> list[ManifestRow]:
        rows: list[ManifestRow] = []
        base_dir = path.parent
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
                image_path = Path(payload["image_path"])
                if not image_path.is_absolute():
                    image_path = base_dir / image_path
                image_path = image_path.resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image path on line {line_number} does not exist: {image_path}"
                    )
                activity = payload["activity_label"]
                education = payload.get("education_label", "general")
                if activity not in ACTIVITY_TO_INDEX:
                    raise ValueError(f"Unknown activity label on line {line_number}: {activity}")
                if education not in EDUCATION_TO_INDEX:
                    raise ValueError(f"Unknown education label on line {line_number}: {education}")
                rows.append(
                    ManifestRow(
                        image_path=image_path,
                        activity_index=ACTIVITY_TO_INDEX[activity],
                        education_index=EDUCATION_TO_INDEX[education],
                    )
                )
        if not rows:
            raise ValueError(f"Manifest contains no rows: {path}")
        return rows
