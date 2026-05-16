# Training Data

Seed screenshots are stored under:

```text
data/screenshots/{activity_label}/{education_label}/image.jpg
```

These generated images are enough to validate the training pipeline. For a useful classifier, add real screenshots captured from your own studying, coding, reading, notes, videos, browser sessions, and distractions.

Run:

```powershell
.\scripts\train_activity_model.ps1
```

The trained checkpoint is written to:

```text
models/activity_classifier_best.pt
```
