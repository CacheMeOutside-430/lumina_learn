from __future__ import annotations

import shutil
from threading import Lock
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from shared.schemas import OcrBlock

from vision.preprocess import prepare_ocr_image

ImageArray = NDArray[np.uint8]


class OcrEngine(Protocol):
    def read(self, image_bgr: ImageArray) -> list[OcrBlock]:
        ...


class EasyOcrEngine:
    def __init__(
        self,
        languages: list[str],
        gpu: bool = True,
        min_confidence: float = 0.25,
        max_side: int = 960,
        roi_enabled: bool = True,
    ) -> None:
        self.languages = languages
        self.gpu = gpu
        self.min_confidence = min_confidence
        self.max_side = max_side
        self.roi_enabled = roi_enabled
        self.reader: Any | None = None
        self._lock = Lock()

    def read(self, image_bgr: ImageArray) -> list[OcrBlock]:
        with self._lock:
            reader = self._reader()
            image_rgb = prepare_ocr_image(
                image_bgr,
                max_side=self.max_side,
                roi_enabled=self.roi_enabled,
            )
            results = reader.readtext(
                image_rgb,
                paragraph=False,
                batch_size=1,
                workers=0,
                canvas_size=max(self.max_side, 320),
                mag_ratio=1.0,
            )
        blocks: list[OcrBlock] = []
        for bbox, text, confidence in results:
            confidence_value = float(confidence)
            if confidence_value >= self.min_confidence and text.strip():
                blocks.append(
                    OcrBlock(
                        text=text.strip(),
                        confidence=confidence_value,
                        bbox=[[float(x), float(y)] for x, y in bbox],
                    )
                )
        return blocks

    def _reader(self) -> Any:
        if self.reader is None:
            import easyocr

            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self.reader


class TesseractOcrEngine:
    def __init__(
        self,
        language: str = "eng",
        min_confidence: float = 0.2,
        max_side: int = 960,
        roi_enabled: bool = True,
    ) -> None:
        self.language = language
        self.min_confidence = min_confidence
        self.max_side = max_side
        self.roi_enabled = roi_enabled
        self.pytesseract: Any | None = None
        self._lock = Lock()

    def read(self, image_bgr: ImageArray) -> list[OcrBlock]:
        with self._lock:
            pytesseract = self._pytesseract()
            image_rgb = prepare_ocr_image(
                image_bgr,
                max_side=self.max_side,
                roi_enabled=self.roi_enabled,
            )
            pil_image = Image.fromarray(image_rgb)
            data = pytesseract.image_to_data(
                pil_image,
                lang=self.language,
                config="--oem 1 --psm 6",
                output_type=pytesseract.Output.DICT,
            )
        blocks: list[OcrBlock] = []
        for index, text in enumerate(data.get("text", [])):
            cleaned = text.strip()
            if not cleaned:
                continue
            raw_conf = data["conf"][index]
            try:
                confidence = max(0.0, float(raw_conf) / 100.0)
            except ValueError:
                confidence = 0.0
            if confidence < self.min_confidence:
                continue
            left = float(data["left"][index])
            top = float(data["top"][index])
            width = float(data["width"][index])
            height = float(data["height"][index])
            blocks.append(
                OcrBlock(
                    text=cleaned,
                    confidence=confidence,
                    bbox=[
                        [left, top],
                        [left + width, top],
                        [left + width, top + height],
                        [left, top + height],
                    ],
                )
            )
        return blocks

    def _pytesseract(self) -> Any:
        if self.pytesseract is None:
            import pytesseract

            self.pytesseract = pytesseract
        return self.pytesseract

    @staticmethod
    def available() -> bool:
        return shutil.which("tesseract") is not None


class RapidOcrEngine:
    def __init__(
        self,
        min_confidence: float = 0.25,
        max_side: int = 960,
        roi_enabled: bool = True,
        threads: int = 4,
    ) -> None:
        self.min_confidence = min_confidence
        self.max_side = max_side
        self.roi_enabled = roi_enabled
        self.threads = threads
        self.reader: Any | None = None
        self._lock = Lock()

    def read(self, image_bgr: ImageArray) -> list[OcrBlock]:
        with self._lock:
            reader = self._reader()
            image_rgb = prepare_ocr_image(
                image_bgr,
                max_side=self.max_side,
                roi_enabled=self.roi_enabled,
            )
            results, _ = reader(image_rgb)
        blocks: list[OcrBlock] = []
        for row in results or []:
            bbox, text, confidence = row
            confidence_value = float(confidence)
            cleaned = str(text).strip()
            if confidence_value >= self.min_confidence and cleaned:
                blocks.append(
                    OcrBlock(
                        text=cleaned,
                        confidence=confidence_value,
                        bbox=[[float(x), float(y)] for x, y in bbox],
                    )
                )
        return blocks

    def _reader(self) -> Any:
        if self.reader is None:
            from rapidocr_onnxruntime import RapidOCR

            self.reader = RapidOCR(
                use_cls=False,
                det_limit_type="max",
                det_limit_side_len=self.max_side,
                intra_op_num_threads=self.threads,
                inter_op_num_threads=1,
            )
        return self.reader

    @staticmethod
    def available() -> bool:
        try:
            from rapidocr_onnxruntime import RapidOCR  # noqa: F401
        except Exception:
            return False
        return True
