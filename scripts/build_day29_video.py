"""Render a two-minute results video from measured outputs, with burned captions."""

import hashlib
from html import escape
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
import textwrap

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs/day29-artifacts"


def text_block(text, x, y, *, width=90, size=28, color="#eaf0f7", line_height=40):
    lines = []
    for paragraph in text.splitlines():
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return "".join(f'<text x="{x}" y="{y + i * line_height}" font-size="{size}" fill="{color}">{escape(line)}</text>' for i, line in enumerate(lines))


def frame(title, subtitle, body, number):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900">'
            '<rect width="1600" height="900" fill="#10151c"/>'
            '<g font-family="Arial, sans-serif">'
            + text_block("AI CHALLENGE  /  ДЕНЬ 29", 80, 80, size=22, color="#85c7a8")
            + text_block(title, 80, 155, size=44, width=65, line_height=55)
            + text_block(subtitle, 80, 220, size=23, color="#aab6c7", width=105)
            + body
            + text_block(f"{number:02d}  ·  Локальная Qwen 3.5 9B  ·  Измеренные результаты", 80, 858, size=20, color="#8796aa")
            + '</g></svg>')


def main():
    raw = (ARTIFACTS / "evaluation.json").read_bytes()
    report = json.loads(raw)
    review = json.loads((ARTIFACTS / "quality-review.json").read_text())
    if review["evaluation_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValueError("Quality review belongs to a different run")
    timing = json.loads((ARTIFACTS / "timing-control.json").read_text())
    if not timing.get("complete") or timing["config"]["profiles"] != report["config"]["profiles"]:
        raise ValueError("Controlled timing is incomplete or uses different profiles")
    if timing["config"]["generation_code_sha256"] != report["config"]["generation_code_sha256"]:
        raise ValueError("Timing and quality runs use different generation code")
    if {k: v["digest"] for k, v in timing["models"].items()} != {k: v["digest"] for k, v in report["models"].items()}:
        raise ValueError("Timing and quality runs use different weights")
    b, o, q = ({**report["summary"][p], "median_seconds": timing["summary"][p]["median_seconds"]} for p in ("baseline", "optimized", "q8"))
    quality = review["summary"]
    def gb(value): return f"{value / 1e9:.2f} ГБ"
    def answer(profile, question=3):
        row = next(r for r in report["runs"] if r["profile"] == profile and r["question_id"] == question and r["repeat"] == 1)
        return re.sub(r"\s*\[DOC:[^\]]+\]", "", row["content"]), row
    before, before_row = answer("baseline")
    after, after_row = answer("optimized")
    body = lambda value: text_block(value, 80, 330, width=88, size=30, line_height=48)
    scenes = [
        ("Оптимизируем Qwen под ответы по книге", "Java Concurrency in Practice · глава 6 · локальный RAG", body(
            "Цель: полные ответы с подтверждёнными цитатами при меньшем расходе памяти.\n\n"
            "18 вопросов × 4 конфигурации × 3 повтора.\n"
            "7 вопросов для настройки, 11 для проверки.\n"
            "Все варианты получают одинаковые фрагменты книги."), 12,
         "День 29. Оптимизируем локальную Qwen для ответов по книге о Java concurrency."),
        ("Сначала проверяем параметры", "Temperature, лимит ответа и окно контекста", body(
            "До: temperature 0 · ответ до 3000 · контекст 32768.\n"
            "После: temperature 0 · ответ до 1000 · контекст 8192.\n\n"
            "Проверены temperature 0.1/0.2, контекст 16384 и лимит 1500.\n"
            "Для длинной памяти контекст может вырасти до 32768.\n"
            "Thinking отключён в обоих вариантах."), 16,
         "Параметры выбраны по калибровочным вопросам. Уменьшение лимита токенов само по себе не доказывает ускорение."),
        ("Сравним один и тот же вопрос", "Что FutureRenderer делает, пока загружаются изображения?",
         '<rect x="70" y="285" width="710" height="495" rx="16" fill="#19222d"/>'
         '<rect x="810" y="285" width="720" height="495" rx="16" fill="#19222d"/>'
         + text_block("ДО · исходный prompt", 95, 330, size=27, color="#aab6c7")
         + text_block(before, 95, 390, width=49, size=24, line_height=36)
         + text_block("ПОСЛЕ · новый prompt", 835, 330, size=27, color="#85c7a8")
         + text_block(after, 835, 390, width=49, size=24, line_height=36), 24,
         "Сохранённые ответы настоящей Qwen. Полная история, цитаты и времена находятся в отчёте."),
        ("Новый prompt прослеживает выполнение", "От задачи до ожидания результата и его использования", body(
            "Отправка задачи → параллельная работа → ожидание → результат.\n\n"
            "Назвать блокирующие методы из фрагментов.\n"
            "Покрыть условия и исходы, убрать повторяющийся фон.\n"
            "Подтвердить каждый тезис выбранной цитатой.\n"
            "Отказаться, если фрагменты не дают ответа."), 16,
         "Prompt настроен под объяснение конкурентного кода, сохраняя проверку источников и формат JSON."),
        ("Скорость и расход памяти", "Прогретая генерация; поиск и интерпретация проверены отдельно", body(
            f"До · Q4: {b['median_seconds']:.2f} с · {gb(b['peak_model_vram_bytes'])}.\n"
            f"После · Q4: {o['median_seconds']:.2f} с · {gb(o['peak_model_vram_bytes'])}.\n"
            f"После · Q8: {q['median_seconds']:.2f} с · {gb(q['peak_model_vram_bytes'])}.\n\n"
            f"Время — контроль: {len(timing['cases'])} вопросов × 3 повтора при стабильном питании.\n"
            "Память — выделение модели из /api/ps, без суммирования с RSS."), 18,
         "Измерены время, токены в секунду, память модели и процессов. На Apple Silicon RSS и Metal-память пересекаются."),
        ("Проверяем качество отдельно от формата", "Ожидаемые факты записаны до контрольного прогона", body(
            f"Полнота: до {quality['baseline']['complete_runs']}/{quality['baseline']['answer_runs']}, "
            f"после {quality['optimized']['complete_runs']}/{quality['optimized']['answer_runs']}.\n"
            f"Верные отказы после: {quality['optimized']['correct_refusals']}/{quality['optimized']['refusal_runs']}.\n\n"
            "Цитаты проверяются дословно; смысл оценён отдельно.\n"
            "Отрицательные вопросы тоже передаются модели с контекстом.\n"
            "Оценка Codex на небольшой выборке требует проверки человеком."), 18,
         "Качество проверено по полноте фактов и смысловой поддержке цитат. Это небольшой набор одной главы."),
        ("Результат можно воспроизвести", "Код, ответы, замеры, интерактивный отчёт и видео", body(
            "RAG_LLM_PROFILE=optimized\n\n"
            "scripts/run_day29_evaluation.py — сравнение конфигураций.\n"
            "scripts/check_day29_application.py — настоящий RAG HTTP-путь.\n"
            "docs/day29-local-llm.md — методика и результаты.\n"
            "docs/day29-artifacts/report.html — просмотр всех ответов."), 16,
         "Видео собрано из измеренных результатов и сохранённых ответов. Это демонстрация отчёта, а не непрерывная запись экрана."),
    ]
    elapsed = 0
    cues = []
    def timestamp(value): return f"{value // 3600:02d}:{value // 60 % 60:02d}:{value % 60:02d},000"
    with TemporaryDirectory(prefix="day29-video-") as temporary:
        work = Path(temporary)
        segments = []
        for index, (title, subtitle, content, seconds, caption) in enumerate(scenes, 1):
            svg = ARTIFACTS / f"scene-{index:02d}.svg"
            png = ARTIFACTS / f"scene-{index:02d}.png"
            svg.write_text(frame(title, subtitle, content, index), encoding="utf-8")
            subprocess.run(["rsvg-convert", "-o", str(png), str(svg)], check=True)
            segment = work / f"{index}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png),
                            "-t", str(seconds), "-r", "24", "-an", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "20", "-pix_fmt", "yuv420p", str(segment)], check=True)
            segments.append(segment)
            cues.append(f"{index}\n{timestamp(elapsed)} --> {timestamp(elapsed + seconds)}\n{caption}\n")
            elapsed += seconds
        manifest = work / "segments.txt"
        manifest.write_text("\n".join(f"file '{p}'" for p in segments))
        target = ARTIFACTS / "day29-demo.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(manifest),
                        "-c", "copy", "-movflags", "+faststart", str(target)], check=True)
    (ARTIFACTS / "day29-demo.srt").write_text("\n".join(cues), encoding="utf-8")
    print(target, f"{elapsed}s")


if __name__ == "__main__":
    main()
