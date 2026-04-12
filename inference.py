import os
import json
import httpx
import asyncio
import textwrap
from typing import List, Dict, Any, Optional

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")
ENV_BASE_URL = "http://localhost:7860"
MAX_STEPS = 15

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert ML dataset cleaning engineer. 
    Goals:
    - Fix text errors (typos, grammar)
    - Correct wrong labels (validate current label before relabeling!)
    - Improve dataset quality, uniqueness, and consistency. Remove exact duplicates (use remove)!
    - Avoid unnecessary deletions, and DO NOT repeat actions.
    
    Always choose the highest-impact action.
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_action(client: OpenAI, obs: Dict[str, Any], history: List[str]) -> Dict[str, Any]:
    history_text = "\n".join(history[-3:]) if history else "None"
    
    user_prompt = f"""
Sample Observation:
{json.dumps(obs, indent=2)}

Previous Step History:
{history_text}

What is the best next action?
Return ONLY valid JSON action object:
{{
  "type": "fix_text" | "relabel" | "remove" | "mark_clean",
  "payload": {{
     "corrected_text": "...",
     "new_label": "...",
     "sample_id": "..." 
  }}
}}
"""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        return json.loads(text.strip())
    except Exception as exc:
        print(f"[DEBUG] Call failed: {exc}", flush=True)
        return {"type": "mark_clean", "payload": {}}

async def run_episode(task_id: str, client: OpenAI) -> None:
    log_start(task=task_id, env="dataset-cleaning-env", model=MODEL_NAME)
    rewards = []
    
    async with httpx.AsyncClient() as http_client:
        try:
            r = await http_client.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30.0)
            state_data = r.json()
            obs = state_data.get("observation", {})
        except Exception as e:
            print(f"[DEBUG] Failed to reset {ENV_BASE_URL}: {e}", flush=True)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            return

        steps_taken = 0
        done = False
        history = []
        score = 0.0
        seen_actions = set()
        stagnant_steps = 0
        prev_reward = 0.0
        
        for step in range(1, MAX_STEPS + 1):
            action = get_action(client, obs, history)
            action_str = json.dumps(action, separators=(",", ":"))
            
            # Fix 1: Duplicate action loop guard
            try:
                payload_str = json.dumps(action.get("payload", {}), sort_keys=True)
            except:
                payload_str = str(action.get("payload", {}))
            action_key = (action.get("type"), payload_str)
            
            if action_key in seen_actions:
                break
            seen_actions.add(action_key)
            
            error = None
            try:
                r = await http_client.post(f"{ENV_BASE_URL}/step", json=action, timeout=30.0)
                res = r.json()
                if r.status_code >= 400:
                    error = res.get("detail", str(r.status_code))
                    done = True
                    reward = 0.0
                else:
                    obs = res.get("observation", {})
                    reward = res.get("reward", 0.0)
                    done = res.get("done", True)
                    info = res.get("info", {})
                    if "score" in info:
                        score = info["score"]
                    elif "score" in res:
                        score = res["score"]
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)
                
            # Fix 2: Stagnation check
            if step > 1 and reward <= prev_reward:
                stagnant_steps += 1
            else:
                stagnant_steps = 0
            prev_reward = reward
                
            rewards.append(reward)
            steps_taken = step
            history.append(f"Step {step}: Action: {action.get('type')}, Payload: {action.get('payload')}, Reward: {reward}")
            
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            
            if done or error or stagnant_steps >= 3:
                break
                
        # If score wasn't explicitly returned from env info, get it from state endpoint
        if score == 0.0:
            try:
                r = await http_client.get(f"{ENV_BASE_URL}/state", timeout=30.0)
                state_info = r.json()
                score = state_info.get("score", 0.0)
            except:
                pass
                
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.5
        
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in ["text-cleaning", "label-correction", "dataset-alignment"]:
        await run_episode(task, client)

if __name__ == "__main__":
    asyncio.run(main())
