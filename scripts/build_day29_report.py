"""Build a self-contained comparison viewer from measured answers and reviewed facts."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs/day29-artifacts"


def main():
    data = (ARTIFACTS / "evaluation.json").read_bytes()
    report = json.loads(data)
    review = json.loads((ARTIFACTS / "quality-review.json").read_text())
    if review["evaluation_sha256"] != hashlib.sha256(data).hexdigest():
        raise ValueError("Quality review is for a different evaluation")
    timing = json.loads((ARTIFACTS / "timing-control.json").read_text())
    if not timing.get("complete") or not timing["config"].get("power_state"):
        raise ValueError("Controlled timing is incomplete")
    if timing["config"]["profiles"] != report["config"]["profiles"] or timing["config"]["generation_code_sha256"] != report["config"]["generation_code_sha256"]:
        raise ValueError("Timing and quality runs use different generation configurations")
    if {k: v["digest"] for k, v in timing["models"].items()} != {k: v["digest"] for k, v in report["models"].items()}:
        raise ValueError("Timing and quality runs use different weights")
    payload = json.dumps({"report": report, "review": review, "timing": timing}, ensure_ascii=False).replace("<", "\\u003c")
    template = '''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>День 29 · Оптимизация локальной Qwen</title>
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif;background:#10151c;color:#edf2f8}*{box-sizing:border-box}body{margin:0}main{max-width:1240px;margin:auto;padding:48px 28px}h1{font-size:38px;letter-spacing:-1px;margin:10px 0 12px}h2{font-size:23px;margin:0 0 20px}.eyebrow{color:#85c7a8;letter-spacing:2px;font-size:12px}.intro{color:#aab6c7;max-width:880px;line-height:1.6}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:30px 0}.card,.panel{border:1px solid #303b49;background:#171f29;border-radius:14px;padding:24px}.card strong{display:block;font-size:28px;margin:10px 0}.card span,.small{color:#aab6c7;font-size:13px}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}th,td{text-align:left;padding:14px 10px;border-bottom:1px solid #303b49}th{color:#aab6c7;font-weight:500}section{margin-top:30px}.controls{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:20px}label{display:flex;flex-direction:column;gap:8px;color:#aab6c7;font-size:13px}select{background:#10151c;color:#edf2f8;border:1px solid #425266;padding:10px;border-radius:7px;max-width:100%}.answers{display:grid;grid-template-columns:1fr 1fr;gap:20px}.answer{white-space:pre-wrap;line-height:1.65;font-size:16px;margin:16px 0}.badge{display:inline-block;padding:5px 9px;border:1px solid #425266;border-radius:6px;color:#85c7a8;font-size:12px}blockquote{border-left:2px solid #85c7a8;padding-left:12px;color:#aab6c7;font-size:13px;line-height:1.5;margin:14px 0}.expected{line-height:1.5;color:#b8c8dd}.note{font-size:13px;line-height:1.65;color:#9ba9bc}a{color:#85c7a8}.scroll{overflow:auto}@media(max-width:780px){main{padding:25px 15px}h1{font-size:30px}.cards,.answers{grid-template-columns:1fr}th,td{padding:12px 6px}}
</style>
<main><div class="eyebrow">AI CHALLENGE · ДЕНЬ 29</div><h1>Локальная Qwen под ответы по книге</h1>
<p class="intro">Оптимизация Qwen 3.5 9B для ответов по Java Concurrency in Practice. Одинаковые фрагменты книги, три повтора, отдельные вопросы для настройки и проверки. На этой странице сохранённые результаты настоящего локального прогона.</p>
<div class="cards" id="cards"></div>
<section class="panel"><h2>Сравнение конфигураций</h2><div class="scroll"><table><thead><tr><th>Вариант</th><th>Контекст / ответ</th><th>Время, медиана</th><th>Токенов/с</th><th>Память модели</th><th>RSS процессов</th><th>Полнота</th><th>Верные отказы</th></tr></thead><tbody id="summary"></tbody></table></div><p class="note" id="timing-note"></p><p class="note">Память и RSS приведены из контрольного замера: выделение модели из /api/ps и пик процессов. На Apple Silicon эти значения пересекаются и не суммируются. Время включает генерацию и проверку цитат; поиск и интерпретация измерены отдельно в сквозном прогоне.</p></section>
<section><h2>Ответы до и после</h2><div class="controls"><label>Вопрос<select id="question" aria-label="Вопрос"></select></label><label>Сравнить с<select id="profile" aria-label="Сравнить с"></select></label><label>Повтор<select id="repeat" aria-label="Повтор"><option>1</option><option>2</option><option>3</option></select></label></div><p class="note">Ответы и цитаты — из полного прогона качества. При наличии показывается время контрольного замера; для остальных вопросов оно помечено как разведочное.</p><p id="expected" class="expected"></p><div class="answers"><article class="panel" id="before"></article><article class="panel" id="after"></article></div></section>
<section class="panel"><h2>Что изменилось</h2><p class="intro">Контекст уменьшен с 32768 до 8192, лимит ответа — с 3000 до 1000 токенов. Temperature 0 и отключённый thinking сохранены после проверки альтернатив. Новый prompt требует объяснять порядок действий, ожидание и результат, подтверждая каждый тезис цитатой. Для длинных обязательных данных контекст может увеличиваться до 32768.</p><p class="note">Точное совпадение цитаты проверяет происхождение, а не смысл. Полноту и соответствие цитат тезисам оценил Codex по заранее записанным ожидаемым фактам. Это небольшой набор вопросов одной главы; результаты требуют проверки человеком и не гарантируют качество на других документах.</p><a href="evaluation.json">Полный прогон JSON</a> · <a href="quality-review.json">Оценка качества JSON</a> · <a href="calibration.json">Подбор параметров JSON</a></section>
</main><script id="data" type="application/json">__PAYLOAD__</script>
<script>
const {report,review,timing}=JSON.parse(document.getElementById('data').textContent);
const names={baseline:'До · Q4',compact:'Параметры · Q4',optimized:'После · Q4',q8:'После · Q8'};
const el=(tag,text)=>{const e=document.createElement(tag);e.textContent=text;return e};
const gb=n=>(n/1e9).toFixed(2)+' ГБ';
const baseline=report.summary.baseline,optimized=report.summary.optimized;
const cards=[['Память модели',gb(optimized.peak_model_vram_bytes),'-'+((1-optimized.peak_model_vram_bytes/baseline.peak_model_vram_bytes)*100).toFixed(1)+'% относительно исходного профиля'],['Полные ответы',review.summary.optimized.complete_runs+'/'+review.summary.optimized.answer_runs,'Каждый ответ сопоставлен с ожидаемыми фактами'],['Генерация',timing.summary.optimized.median_seconds.toFixed(2)+' с','Контрольный замер при одинаковом режиме питания']];
for(const [label,value,description] of cards){const c=el('div');c.className='card';c.append(el('span',label),el('strong',value),el('span',description));document.getElementById('cards').append(c)}
for(const [name,s] of Object.entries(report.summary)){const t=timing.summary[name],p=report.config.profiles[name],q=review.summary[name],tr=el('tr');for(const value of [names[name],p.num_ctx+' / '+p.max_tokens,t.median_seconds.toFixed(2)+' с',t.median_tokens_per_second,gb(t.peak_model_vram_bytes),gb(t.peak_ollama_rss_bytes),q.complete_runs+'/'+q.answer_runs,q.correct_refusals+'/'+q.refusal_runs])tr.append(el('td',value));document.getElementById('summary').append(tr)}
document.getElementById('timing-note').textContent='Время и ресурсы: '+timing.cases.length+' вопросов × '+timing.config.repeats+' повтора на вариант; '+timing.config.power_state.source+', power mode '+timing.config.power_state.power_mode+'. Качество: полный набор из 18 вопросов. Задержки полного прогона менялись вместе с условиями машины и не используются для вывода об ускорении.';
const question=document.getElementById('question'),profile=document.getElementById('profile'),repeat=document.getElementById('repeat');
for(const c of report.cases){const o=el('option',c.id+'. '+c.question);o.value=c.id;question.append(o)}
for(const name of ['compact','optimized','q8']){const o=el('option',names[name]);o.value=name;profile.append(o)}
question.value='3';profile.value='optimized';
function renderAnswer(id,name){const box=document.getElementById(id);box.replaceChildren();const row=report.runs.find(r=>r.profile===name&&r.question_id===Number(question.value)&&r.repeat===Number(repeat.value));box.append(el('h2',names[name]));if(!row){box.append(el('p','Результат отсутствует'));return}const measured=timing.runs.find(r=>r.profile===name&&r.question_id===row.question_id&&r.repeat===row.repeat);const badge=el('span',(measured?'Контроль · ':'Разведочный · ')+(measured||row).elapsed_seconds.toFixed(2)+' с · '+row.status);badge.className='badge';box.append(badge);const a=el('p',row.content);a.className='answer';box.append(a);for(const c of row.citations)box.append(el('blockquote',c.quote));const grade=review.runs.find(r=>r.profile===name&&r.question_id===row.question_id&&r.repeat===row.repeat);if(grade){const p=el('p',grade.note);p.className='note';box.append(p)}}
function render(){const c=report.cases.find(c=>c.id===Number(question.value));document.getElementById('expected').textContent='Ожидаемые факты: '+c.expected;renderAnswer('before','baseline');renderAnswer('after',profile.value)}
for(const control of [question,profile,repeat])control.addEventListener('change',render);render();
</script></html>'''
    target = ARTIFACTS / "report.html"
    target.write_text(template.replace("__PAYLOAD__", payload), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
