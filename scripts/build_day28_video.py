"""Edit actual browser screenshots into an MP4 with burned Russian captions."""

import json
import hashlib
from html import escape
from pathlib import Path
import subprocess
import tempfile
import textwrap


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs/day28-artifacts"


def main() -> None:
    evaluation = (ARTIFACTS / "evaluation.json").read_bytes()
    summary = json.loads(evaluation)["summary"]
    review = json.loads((ARTIFACTS / "quality-review.json").read_text())
    if review["evaluation_sha256"] != hashlib.sha256(evaluation).hexdigest():
        raise ValueError("Смысловая оценка относится к другому прогону; обновите quality-review.json")
    local = summary["ollama"]["median_generation_seconds"]
    cloud = summary["deepseek"]["median_generation_seconds"]
    scenes = [
        ("01-local-ready.jpg", "День 28. Индекс книги из W4 и локальная Qwen через Ollama.", 4),
        ("02-local-generating.jpg", "Вопрос проходит через локальные E5, FAISS и reranker. Ответ генерирует Qwen.", 4),
        ("03-local-answer.jpg", "Локальный ответ с проверенной цитатой, страницей PDF и временем каждого этапа.", 6),
        ("04-cloud-generating.jpg", "Переключаем модель на DeepSeek и повторяем тот же вопрос.", 4),
        ("05-cloud-answer.jpg", "Ответ DeepSeek. В сравнительном прогоне обе модели получают одинаковые фрагменты.", 6),
        ("06-negative-answer.jpg", "Для StructuredTaskScope подтверждения нет: RAG отказывает без выдуманных источников.", 5),
        ("07-offline-memory.jpg", "Отдельный backend без облачного ключа; внешние соединения запрещены. Память пережила перезапуск.", 5),
        ("08-offline-generating.jpg", "Новый вопрос в офлайн-процессе. Используются только локальные веса и индекс.", 4),
        ("09-offline-answer.jpg", "Настоящая Qwen ответила с цитатами при запрещённых внешних соединениях backend.", 6),
        ("09-offline-answer.jpg", f"21 генерация на модель: медиана Qwen {local:.2f} с, DeepSeek {cloud:.2f} с. Ошибок провайдеров и проверки цитат нет.", 6),
        ("09-offline-answer.jpg", f"Полнота по ожидаемым фактам: Qwen {review['summary']['ollama']['complete_runs']}/21, DeepSeek {review['summary']['deepseek']['complete_runs']}/21. Статусы совпали в трёх повторах.", 6),
    ]
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "json",
        str(ARTIFACTS / scenes[0][0]),
    ]))
    width, height = (probe["streams"][0][key] for key in ("width", "height"))
    width, height = width + width % 2, height + height % 2
    elapsed, cues = 0, []
    def timestamp(value):
        return f"{value // 3600:02d}:{value // 60 % 60:02d}:{value % 60:02d},000"
    with tempfile.TemporaryDirectory(prefix="day28-video-") as temporary:
        work = Path(temporary)
        segments = []
        for index, (filename, caption, seconds) in enumerate(scenes):
            svg = work / f"caption-{index}.svg"
            caption_png = work / f"caption-{index}.png"
            lines = textwrap.wrap(caption, width=85)
            text = "".join(f'<text x="{width/2}" y="{32+line*28}" text-anchor="middle" '
                           f'font-family="Arial, sans-serif" font-size="21" fill="white">{escape(value)}</text>'
                           for line, value in enumerate(lines))
            svg.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="100">'
                           f'<rect width="100%" height="100%" fill="#0c121d"/>{text}</svg>', encoding="utf-8")
            subprocess.run(["rsvg-convert", "-o", str(caption_png), str(svg)], check=True)
            output = work / f"scene-{index}.mp4"
            filters = (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x0c121d,"
                "pad=iw:ih+100:0:0:color=0x0c121d[screen];"
                f"[screen][1:v]overlay=0:{height}[out]"
            )
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(ARTIFACTS / filename),
                "-loop", "1", "-i", str(caption_png), "-t", str(seconds),
                "-filter_complex", "[0:v]" + filters, "-map", "[out]", "-r", "24", "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(output),
            ], check=True)
            segments.append(output)
            cues.append(f"{index+1}\n{timestamp(elapsed)} --> {timestamp(elapsed+seconds)}\n{caption}\n")
            elapsed += seconds
        manifest = work / "scenes.txt"
        manifest.write_text("\n".join(f"file '{path}'" for path in segments), encoding="utf-8")
        target = ARTIFACTS / "day28-demo.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(manifest),
            "-c", "copy", "-movflags", "+faststart", str(target),
        ], check=True)
        (ARTIFACTS / "day28-demo.srt").write_text("\n".join(cues), encoding="utf-8")
        print(target)


if __name__ == "__main__":
    main()
