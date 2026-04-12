from copy import deepcopy
from tasks.scoring import clamp_open_unit_interval

def get_initial_obs_task3():
    dataset = [
        {"sample_id": "3a", "text": "This is great!", "label": "positive", "issues": [], "confidence": 0.9},
        {"sample_id": "3b", "text": "This is great!", "label": "negative", "issues": ["duplicate", "wrong_label"], "confidence": 0.4},
        {"sample_id": "3c", "text": "Horrible item.", "label": "negative", "issues": [], "confidence": 0.9},
        {"sample_id": "3d", "text": "I really love it.", "label": "negative", "issues": ["wrong_label"], "confidence": 0.3}
    ]
    obs = {
        "sample_id": "dataset-view",
        "text": "Multi-sample view",
        "label": "mixed",
        "issues": ["duplicate", "wrong_label", "distribution_imbalance"],
        "confidence": 0.5,
        "step": 0,
        "history": [],
        "dataset": dataset
    }
    return obs, dataset

def step_task3(state, action):
    obs = deepcopy(state["obs"])
    reward = 0.0
    done = False
    if action.type == "remove":
        target = action.payload.get("sample_id", None)
        if target and target not in [s["sample_id"] for s in obs["dataset"]]:
            return obs, float(-0.1), False, {"error": "invalid_sample_id"}
        if target:
            original_len = len(obs["dataset"])
            obs["dataset"] = [s for s in obs["dataset"] if s["sample_id"] != target]
            if len(obs["dataset"]) < original_len:
                reward += 0.3
            else:
                reward -= 0.1
    elif action.type == "relabel":
        target = action.payload.get("sample_id", None)
        if target and target not in [s["sample_id"] for s in obs["dataset"]]:
            return obs, float(-0.1), False, {"error": "invalid_sample_id"}
        new_label = action.payload.get("new_label", None)
        if target and new_label:
            found = False
            for s in obs["dataset"]:
                if s["sample_id"] == target:
                    s["label"] = new_label
                    found = True
                    reward += 0.2
            if not found:
                reward -= 0.1
    elif action.type == "fix_text":
        reward -= 0.1 # Should not happen here
    elif action.type == "mark_clean":
        done = True
        
    obs["step"] += 1
    obs["history"].append(f"{action.type}: {action.payload}")
    
    return obs, float(reward), done, {}

def grade_task3(state):
    dataset = state["obs"]["dataset"]
    score = 0.0
    
    # 1. Uniqueness
    texts = [s["text"] for s in dataset]
    if len(texts) == len(set(texts)):
        score += 0.3
        
    # 2. Label consistency 
    consistency_ok = True
    for s in dataset:
        if s["text"] == "This is great!" and s["label"] != "positive":
            consistency_ok = False
        if s["text"] == "Horrible item." and s["label"] != "negative":
            consistency_ok = False
        if s["text"] == "I really love it." and s["label"] != "positive":
            consistency_ok = False
            
    if consistency_ok:
        score += 0.4
        
    # 3. Class balance
    pos = sum(1 for s in dataset if s["label"] == "positive")
    neg = sum(1 for s in dataset if s["label"] == "negative")
    if pos > 0 and neg > 0 and abs(pos - neg) <= 1:
        score += 0.3
        
    return round(clamp_open_unit_interval(score), 3)
