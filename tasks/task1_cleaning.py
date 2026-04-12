from copy import deepcopy
import Levenshtein
from tasks.scoring import clamp_open_unit_interval

def get_initial_obs_task1():
    dataset = [{
        "sample_id": "1",
        "text": "Ths prodct is absolutly amazng",
        "label": "positive",
        "issues": ["typo"],
        "confidence": 0.95
    }]
    obs = {
        "sample_id": dataset[0]["sample_id"],
        "text": dataset[0]["text"],
        "label": dataset[0]["label"],
        "issues": dataset[0]["issues"],
        "confidence": dataset[0]["confidence"],
        "step": 0,
        "history": [],
        "dataset": None
    }
    return obs, dataset

def step_task1(state, action):
    obs = deepcopy(state["obs"])
    reward = 0.0
    done = False
    
    prev_text = obs["text"]
    reference = "This product is absolutely amazing"
    
    if action.type == "fix_text":
        new_text = action.payload.get("corrected_text", obs["text"])
        
        # Calculate improvement
        old_dist = Levenshtein.distance(prev_text, reference)
        new_dist = Levenshtein.distance(new_text, reference)
        
        if new_dist < old_dist:
            reward += (old_dist - new_dist) * 0.2
            obs["issues"] = []
        elif new_dist > old_dist:
            reward -= 0.2
            
        obs["text"] = new_text
            
    elif action.type == "mark_clean":
        if "typo" not in obs["issues"] or len(obs["issues"]) == 0:
            reward += 0.5
        else:
            reward -= 0.5
        done = True
        
    elif action.type == "remove":
        reward -= 0.3
        
    obs["step"] += 1
    obs["history"].append(f"{action.type}: {action.payload}")
    
    return obs, float(reward), done, {}

def grade_task1(state):
    final_text = state["obs"]["text"]
    reference = "This product is absolutely amazing"
    
    score = 0.0
    dist = Levenshtein.distance(final_text, reference)
    if dist < Levenshtein.distance("Ths prodct is absolutly amazng", reference):
        score += 0.5
    if dist <= 2:
        score += 0.5
        
    return round(clamp_open_unit_interval(score), 3)
