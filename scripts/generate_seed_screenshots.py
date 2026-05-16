from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LABEL_SPECS = [
    ("coding", "programming", "IDE: Python project", ["main.py", "trainer.py", "README.md"]),
    ("reading", "research", "Research PDF", ["Abstract", "Method", "Results"]),
    ("writing", "note_taking", "Study notes", ["Summary", "Questions", "Next steps"]),
    ("browser", "general", "Browser search", ["docs", "tutorial", "reference"]),
    ("video", "general", "Lecture video", ["Timeline", "Transcript", "Comments"]),
    ("messaging", "general", "Chat window", ["Messages", "Thread", "Replies"]),
    ("gaming", "general", "Game screen", ["Inventory", "Score", "Map"]),
    ("idle", "general", "Desktop idle", ["Calendar", "Files", "Clock"]),
    ("unknown", "general", "Mixed desktop", ["Window A", "Panel B", "Widget C"]),
]


PALETTES = [
    ("#101418", "#1f2937", "#10b981", "#e5e7eb"),
    ("#111827", "#273449", "#38bdf8", "#f8fafc"),
    ("#171717", "#292524", "#f59e0b", "#fafafa"),
    ("#0f172a", "#334155", "#a3e635", "#f1f5f9"),
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = load_font(20)
FONT_SMALL = load_font(15)
FONT_MONO = load_font(17)
FONT_BIG = load_font(38)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate seed screenshot-style training images.")
    parser.add_argument("--output", type=Path, default=Path("data/screenshots"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    total = 0
    for spec_index, (activity, education, title, terms) in enumerate(LABEL_SPECS):
        folder = args.output / activity / education
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(args.count):
            rng = random.Random((spec_index + 1) * 10_000 + index)
            image = render_scene(
                activity=activity,
                education=education,
                title=title,
                terms=terms,
                width=args.width,
                height=args.height,
                rng=rng,
                variant=index,
            )
            image.save(folder / f"{activity}_{education}_{index:03d}.jpg", quality=88, optimize=True)
            total += 1
    print({"output": str(args.output.resolve()), "images": total}, flush=True)


def render_scene(
    activity: str,
    education: str,
    title: str,
    terms: list[str],
    width: int,
    height: int,
    rng: random.Random,
    variant: int,
) -> Image.Image:
    bg, panel, accent, text = rng.choice(PALETTES)
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    draw_top_bar(draw, width, panel, accent, title, text)

    if activity == "coding":
        draw_ide(draw, width, height, panel, accent, text, rng, variant)
    elif activity == "reading":
        draw_pdf(draw, width, height, panel, accent, text, rng)
    elif activity == "writing":
        draw_notes(draw, width, height, panel, accent, text, rng)
    elif activity == "browser":
        draw_browser(draw, width, height, panel, accent, text, rng)
    elif activity == "video":
        draw_video(draw, width, height, panel, accent, text, rng)
    elif activity == "messaging":
        draw_messages(draw, width, height, panel, accent, text, rng)
    elif activity == "gaming":
        draw_game(draw, width, height, panel, accent, text, rng)
    elif activity == "idle":
        draw_idle(draw, width, height, panel, accent, text, rng)
    else:
        draw_unknown(draw, width, height, panel, accent, text, rng)

    footer = f"activity={activity}   context={education}   {' | '.join(terms)}"
    draw.text((34, height - 34), footer, fill=text, font=FONT_SMALL)
    add_noise(draw, width, height, rng)
    return image


def draw_top_bar(draw: ImageDraw.ImageDraw, width: int, panel: str, accent: str, title: str, text: str) -> None:
    draw.rounded_rectangle((20, 18, width - 20, 70), radius=10, fill=panel)
    for offset, color in enumerate(("#ef4444", "#f59e0b", "#22c55e")):
        draw.ellipse((40 + offset * 28, 34, 54 + offset * 28, 48), fill=color)
    draw.text((150, 32), title, fill=text, font=FONT)
    draw.rounded_rectangle((width - 270, 32, width - 44, 56), radius=7, outline=accent, width=2)
    draw.text((width - 255, 35), "Lumina local capture", fill=text, font=FONT_SMALL)


def draw_ide(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    panel: str,
    accent: str,
    text: str,
    rng: random.Random,
    variant: int,
) -> None:
    draw.rounded_rectangle((30, 92, 260, height - 58), radius=8, fill="#111827")
    draw.rounded_rectangle((282, 92, width - 30, height - 58), radius=8, fill="#0b1120")
    files = ["app.py", "model.py", "dataset.py", "trainer.py", "config.yaml"]
    for index, file_name in enumerate(files):
        y = 124 + index * 42
        fill = accent if index == variant % len(files) else text
        draw.text((58, y), file_name, fill=fill, font=FONT_SMALL)
    code = [
        "async def process_frame(frame: np.ndarray) -> FrameAnalysis:",
        "    ocr_blocks = await vision.read_text(frame)",
        "    activity = classifier.predict(frame)",
        "    embedding = await embeddings.embed_text(ocr_text)",
        "    suggestion = await tutor.suggest(context)",
        "    return analysis",
    ]
    for index in range(26):
        line = code[index % len(code)]
        y = 118 + index * 20
        draw.text((318, y), f"{index + 1:>2}  {line}", fill=text if index % 3 else accent, font=FONT_MONO)


def draw_pdf(draw: ImageDraw.ImageDraw, width: int, height: int, panel: str, accent: str, text: str, rng: random.Random) -> None:
    draw.rounded_rectangle((88, 96, width - 88, height - 60), radius=8, fill="#f8fafc")
    draw.text((142, 124), "Attention and learning behavior in digital study environments", fill="#111827", font=FONT)
    for line in range(17):
        y = 176 + line * 24
        x2 = width - 160 - rng.randint(0, 220)
        draw.rounded_rectangle((142, y, x2, y + 10), radius=3, fill="#334155")
    draw.rectangle((142, 590, width - 142, 596), fill=accent)
    draw.text((142, 612), "Highlight: retrieval practice improves retention after spaced review.", fill="#111827", font=FONT_SMALL)


def draw_notes(draw: ImageDraw.ImageDraw, width: int, height: int, panel: str, accent: str, text: str, rng: random.Random) -> None:
    draw.rounded_rectangle((80, 104, width - 80, height - 64), radius=8, fill="#fff7ed")
    draw.text((120, 130), "Study Notes", fill="#111827", font=FONT_BIG)
    headings = ["Concept", "Example", "Question", "Review"]
    for index, heading in enumerate(headings):
        y = 205 + index * 92
        draw.text((126, y), heading, fill="#7c2d12", font=FONT)
        for line in range(2):
            draw.line((250, y + 12 + line * 24, width - 150 - rng.randint(0, 200), y + 12 + line * 24), fill="#475569", width=3)
    draw.rounded_rectangle((width - 340, 126, width - 130, 184), radius=8, fill=accent)
    draw.text((width - 316, 144), "next review: today", fill="#111827", font=FONT_SMALL)


def draw_browser(draw: ImageDraw.ImageDraw, width: int, height: int, panel: str, accent: str, text: str, rng: random.Random) -> None:
    draw.rounded_rectangle((48, 104, width - 48, height - 62), radius=8, fill="#f8fafc")
    draw.rounded_rectangle((88, 130, width - 88, 172), radius=18, outline="#64748b", width=2)
    draw.text((118, 141), "how to solve dynamic programming recurrence", fill="#0f172a", font=FONT_SMALL)
    for index, title in enumerate(["Official documentation", "Interactive tutorial", "Research article", "Forum discussion"]):
        y = 218 + index * 92
        draw.text((104, y), title, fill="#1d4ed8", font=FONT)
        draw.text((104, y + 30), "A concise explanation with examples, diagrams, and step-by-step reasoning.", fill="#334155", font=FONT_SMALL)
        draw.line((104, y + 64, width - 130 - rng.randint(0, 200), y + 64), fill=accent, width=2)


def draw_video(draw: ImageDraw.ImageDraw, width: int, height: int, panel: str, accent: str, text: str, rng: random.Random) -> None:
    draw.rounded_rectangle((60, 104, width - 360, height - 92), radius=8, fill="#020617")
    draw.polygon([(width // 2 - 85, 325), (width // 2 - 85, 245), (width // 2, 285)], fill=accent)
    draw.rounded_rectangle((92, height - 134, width - 394, height - 118), radius=6, fill="#334155")
    draw.rounded_rectangle((92, height - 134, 420 + rng.randint(0, 180), height - 118), radius=6, fill=accent)
    draw.rounded_rectangle((width - 330, 104, width - 48, height - 92), radius=8, fill=panel)
    for index in range(8):
        y = 132 + index * 58
        draw.rectangle((width - 304, y, width - 254, y + 36), fill="#475569")
        draw.text((width - 238, y + 8), f"Lecture segment {index + 1}", fill=text, font=FONT_SMALL)


def draw_messages(draw: ImageDraw.ImageDraw, width: int, height: int, panel: str, accent: str, text: str, rng: random.Random) -> None:
    draw.rounded_rectangle((74, 102, width - 74, height - 70), radius=8, fill=panel)
    for index in range(9):
        left = 130 if index % 2 == 0 else width - 550
        right = left + 420 + rng.randint(-40, 70)
        y = 138 + index * 54
        fill = "#1f2937" if index % 2 == 0 else accent
        draw.rounded_rectangle((left, y, right, y + 36), radius=16, fill=fill)
        copy = "Can you send the notes?" if index % 2 == 0 else "One sec, checking now."
        draw.text((left + 18, y + 9), copy, fill=text if index % 2 == 0 else "#111827", font=FONT_SMALL)


def draw_game(draw: ImageDraw.ImageDraw, width: int, height: int, panel: str, accent: str, text: str, rng: random.Random) -> None:
    draw.rectangle((34, 88, width - 34, height - 56), fill="#14532d")
    for _index in range(22):
        x = rng.randint(50, width - 70)
        y = rng.randint(110, height - 90)
        draw.rectangle((x, y, x + 32, y + 32), fill=rng.choice(["#166534", "#15803d", "#365314"]))
    draw.rounded_rectangle((50, 104, 306, 174), radius=8, fill="#0f172a")
    draw.text((72, 124), "Score 12840   Quest active", fill=text, font=FONT)
    draw.rounded_rectangle((width - 320, 104, width - 60, 300), radius=8, fill="#0f172a")
    for index in range(8):
        x = width - 292 + (index % 4) * 58
        y = 132 + (index // 4) * 64
        draw.rectangle((x, y, x + 42, y + 42), outline=accent, width=2)


def draw_idle(draw: ImageDraw.ImageDraw, width: int, height: int, panel: str, accent: str, text: str, rng: random.Random) -> None:
    for index in range(12):
        x = 80 + (index % 6) * 160
        y = 130 + (index // 6) * 150
        draw.rounded_rectangle((x, y, x + 92, y + 76), radius=8, fill=panel)
        draw.text((x + 12, y + 92), f"file_{index + 1}", fill=text, font=FONT_SMALL)
    draw.rounded_rectangle((width - 340, height - 240, width - 80, height - 88), radius=8, fill=panel)
    draw.text((width - 294, height - 198), "14:25", fill=accent, font=FONT_BIG)
    draw.text((width - 284, height - 150), "No active study app", fill=text, font=FONT_SMALL)


def draw_unknown(draw: ImageDraw.ImageDraw, width: int, height: int, panel: str, accent: str, text: str, rng: random.Random) -> None:
    for index in range(7):
        x = rng.randint(60, width - 460)
        y = rng.randint(108, height - 270)
        w = rng.randint(250, 460)
        h = rng.randint(120, 240)
        draw.rounded_rectangle((x, y, x + w, y + h), radius=8, fill=rng.choice([panel, "#1e293b", "#27272a"]))
        draw.text((x + 18, y + 18), f"Mixed window {index + 1}", fill=text, font=FONT_SMALL)
        draw.line((x + 18, y + 54, x + w - 24, y + 54), fill=accent, width=2)


def add_noise(draw: ImageDraw.ImageDraw, width: int, height: int, rng: random.Random) -> None:
    for _ in range(80):
        x = rng.randint(0, width - 1)
        y = rng.randint(0, height - 1)
        shade = int(90 + 60 * math.sin(x * 0.01 + y * 0.013))
        draw.point((x, y), fill=(shade, shade, shade))


if __name__ == "__main__":
    main()
