"""Run against an isolated localhost server started WITHOUT DEEPSEEK_API_KEY."""

import argparse
import json
import time
from pathlib import Path
import httpx

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-url', default='http://127.0.0.1:8765')
parser.add_argument('--output', type=Path, default=Path('docs/day27-artifacts/local-smoke.json'))
args = parser.parse_args()
base = args.base_url
args.output.parent.mkdir(parents=True, exist_ok=True)
records = []
with httpx.Client(base_url=base, timeout=300, trust_env=False) as client:
    def call(method, url, payload=None):
        started = time.monotonic()
        response = client.request(method, url, json=payload)
        data = response.json()
        records.append({'method': method, 'path': url, 'elapsed_seconds': round(time.monotonic()-started, 2), 'status': response.status_code, 'response': data})
        args.output.write_text(json.dumps({'cloud_key_present': records[0]['response'].get('deepseek_configured'), 'records': records}, ensure_ascii=False, indent=2))
        response.raise_for_status()
        print(method, url, response.status_code, records[-1]['elapsed_seconds'], flush=True)
        return data
    assert not call('GET', '/api/health')['deepseek_configured']
    settings = call('GET', '/api/invariants')
    settings.update(emoji_enabled=False, uppercase_enabled=False, sentence_limit_enabled=False)
    call('PUT', '/api/invariants', settings)
    profile = call('POST', '/api/profiles/auto')
    session = call('POST', '/api/chat/sessions', {'profile_id': profile['id']})
    url = '/api/chat/sessions/' + session['id']
    for message in ['Объясни, что такое локальная LLM, в двух предложениях.',
                    'Меня зовут Демо, я изучаю Python и пока новичок.',
                    'Хочу краткие ответы обычным текстом.',
                    'Дружелюбный тон. Без сложного жаргона.']:
        result = call('POST', url+'/messages', {'content': message, 'provider': 'ollama'})
    assert result['assistant_message']['provider'] == 'ollama'
    ready = call('GET', '/api/profiles')
    assert next(p for p in ready if p['id'] == profile['id'])['onboarding_complete']
    call('POST', url+'/messages', {'content': 'Запомни: мой учебный проект называется Маяк.'})
    memory = call('GET', '/api/memory?session_id='+session['id'])
    assert any('Маяк' in e['content'] for e in memory['long_term']+memory['working'])
    call('POST', url+'/messages', {'content': 'Как называется мой учебный проект?'})
    call('POST', url+'/task', {'task': 'Напиши одну короткую строку приветствия для участника курса. Один шаг плана. Критерий: строка содержит слово привет.'})
    for i in range(12):
        task = call('GET', url)['task']
        if task['state'] == 'done':
            break
        action = 'approve' if task['expected_action'] == 'approve_plan' else task['expected_action']
        call('POST', url+'/task/actions', {'action': action, 'revision': task['revision']})
    assert call('GET', url)['task']['state'] == 'done'
    print('LOCAL_SMOKE_COMPLETE', flush=True)
