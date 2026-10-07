"""Build an edited demo from actual UI screenshots; requires ffmpeg."""

from pathlib import Path
import subprocess
import tempfile


root = Path(__file__).resolve().parents[1]
artifacts = root / "docs/day27-artifacts"
scenes = [
    ("01-local-ready.jpg", "День 27: локальная Qwen и DeepSeek в одном приложении", 4),
    ("02-local-question.jpg", "Выбираем локальную модель и отправляем вопрос", 3),
    ("03-local-generating.jpg", "Приложение отправляет запрос в Ollama на этом компьютере", 3),
    ("05-local-answer.jpg", "Ответ локальной Qwen; следующий запрос выбран для DeepSeek", 5),
    ("06-cloud-generating.jpg", "Запрос к DeepSeek в том же чате", 3),
    ("07-cloud-answer.jpg", "Обе модели ответили: источник подписан у каждого ответа", 5),
    ("08-queue-pinned.jpg", "Выбор сменился, а сообщение в очереди осталось локальным", 5),
    ("09-queue-local-answers.jpg", "Локальная модель сохранила контекст и обработала очередь", 5),
    ("10-final.jpg", "Выбор и история сохранились после обновления страницы", 4),
]

with tempfile.TemporaryDirectory(prefix="day27-video-") as temporary:
    work = Path(temporary)
    segments = []
    for index, (filename, caption, duration) in enumerate(scenes):
        segment = work / f"scene-{index}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-loop", "1",
            "-i", str(artifacts / filename), "-t", str(duration),
            "-r", "24", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(segment),
        ], check=True)
        segments.append(segment)
    manifest = work / "scenes.txt"
    manifest.write_text("\n".join(f"file '{segment}'" for segment in segments))
    merged = work / "merged.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(manifest), "-c", "copy", str(merged),
    ], check=True)
    subtitles = artifacts / "day27-demo.srt"
    def timestamp(seconds):
        return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d},000"
    elapsed = 0
    cues = []
    for index, (_, caption, duration) in enumerate(scenes, 1):
        cues.append(f"{index}\n{timestamp(elapsed)} --> {timestamp(elapsed + duration)}\n{caption}\n")
        elapsed += duration
    subtitles.write_text("\n".join(cues))
    output = artifacts / "day27-demo.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(merged), "-i", str(subtitles),
        "-map", "0:v", "-map", "1:s", "-c:v", "copy", "-c:s", "mov_text",
        "-metadata:s:s:0", "language=rus", "-disposition:s:0", "default",
        "-movflags", "+faststart", str(output),
    ], check=True)
    print(output)
