# ML Pipeline

## Activity Classifier

The CNN classifier is a compact residual-style network optimized for screenshots. It predicts:

- Activity labels: coding, reading, writing, video, browser, messaging, gaming, idle, unknown.
- Educational context labels: programming, math, science, language, research, exam_prep, note_taking, general.

Training uses:

- JSONL manifests for reproducible datasets.
- Image augmentations suitable for UI screenshots.
- Mixed precision when CUDA is available.
- Best-checkpoint persistence by validation loss.
- Batch inference support for realtime and offline analytics.

## OCR

EasyOCR is the default engine because it works without an external Tesseract binary. Tesseract is supported for deployments that already standardize around it. OCR output is confidence-filtered and passed to downstream embedding and reasoning services.

## Embeddings and Memory

Text from OCR, foreground app metadata, and classifier labels are embedded with a sentence-transformer model. Memories are stored in Chroma or FAISS and keyed by session/user metadata.

## Reasoning

Local transformer reasoning uses Hugging Face `transformers` text2text models. The provider interface supports future cloud inference without changing orchestration code.

## Fine-Tuning Hooks

Training modules expose explicit checkpoint, optimizer, scheduler, and manifest boundaries so teams can add:

- User-personalized classifier heads.
- Distillation from larger multimodal teachers.
- ONNX/TensorRT export for desktop inference.
- Federated or privacy-preserving gradient aggregation.
