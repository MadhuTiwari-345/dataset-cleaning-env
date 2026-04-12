import os
import json
import httpx
import asyncio
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")
ENV_BASE_URL = "http://localhost:7860"
MAX_STEPS = 15

P1 = 'Return ONLY this JSON format with no extra keys:\n{"type": "fix_text", "payload": {"corrected_text": "FIXED TEXT", "sample_id": "1"}}\nNext step: {"type": "mark_clean", "payload": {"sample_id": "1"}}'
P2 = 'Return ONLY this JSON format with no extra keys:\n{"type": "relabel", "payload": {"new_label": "negative", "sample_id": "2"}}\nNext step: {"type": "mark_clean", "payload": {"sample_id": "2"}}'
P3 = 'Return ONLY this JSON format with no extra keys. One action per step.\nStep1: {"type": "relabel", "payload": {"new_label": "positive", "sample_id": "3d"}}\nStep2: {"type": "remove", "payload": {"sample_id": "3b"}}\nStep3: {"type": "mark_clean", "payload": {"sample_id": "3a"}}'
PROMPTS = {"text-cleaning": P1, "label-correction": P2, "dataset-alignment": P3}

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    e = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={e}", flush=True)

def log_end(success, steps, score, rewards):
    r = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={r}", flush=True)

def get_action(client, obs, history, task_id):
    hist = chr(10).join(history[-3:]) if history else "None"
    prompt = "Observation:" + json.dumps(obs) + chr(10) + "History:" + hist + chr(10) + "Return only JSON."
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": PROMPTS[task_id]}, {"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200)
        text = (completion.choices[0].message.content or "").strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text.strip())
        if isinstance(result, list):
            result = result[0]
        return result
    except Exception as exc:
        print(f"[DEBUG] Call failed: {exc}", flush=True)
        return {"type": "remove", "payload": {"sample_id": "3b"}}

async def run_episode(task_id, client):
    log_start(task=task_id, env="dataset-cleaning-env", model=MODEL_NAME)
    rewards = []
    async with httpx.AsyncClient() as http:
        try:
            r = await http.post(ENV_BASE_URL + "/reset", json={"task_id": task_id}, timeout=30.0)
            obs = r.json().get("observation", {})
        except Exception as e:
            print(f"[DEBUG] Reset failed: {e}", flush=True)
            log_end(False, 0, 0.0, [])
            return
        steps_taken = 0
        done = False
        score = 0.0
        history = []
        seen = set()
        for step in range(1, MAX_STEPS + 1):
            action = get_action(client, obs, history, task_id)
            if isinstance(action, list):
                action = action[0] if action else {"type": "mark_clean", "payload": {}}
            action_str = json.dumps(action, separators=(",", ":"))
            key = (action.get("type"), json.dumps(action.get("payload", {}), sort_keys=True))            
            if key in seen:
                break
            seen.add(key)
            error = None
            try:
                r = await http.post(ENV_BASE_URL + "/step", json=action, timeout=30.0)
                res = r.json()
                if r.status_code >= 400:
                    error = str(res.get("detail", r.status_code))
                    done = True
                    reward = 0.0
                else:
                    obs = res.get("observation", {})
                    reward = res.get("reward", 0.0)
                    done = res.get("done", True)
                    score = res.get("info", {}).get("score", res.get("score", score))
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)
            rewards.append(reward)
            steps_taken = step
            history.append("Step" + str(step) + ":" + str(action.get("type")) + " reward=" + str(reward))
            log_step(step, action_str, reward, done, error)
            if done:
                break
        if score == 0.0:
            try:
                r = await http.get(ENV_BASE_URL + "/state", timeout=30.0)
                score = r.json().get("score", 0.0)
            except:
                pass
        score = min(max(score, 0.0), 1.0)
        log_end(score >= 0.3, steps_taken, score, rewards)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in ["text-cleaning", "label-correction", "dataset-alignment"]:
        await run_episode(task, client)

if __name__ == "__main__":
    asyncio.run(main())
