from copy import deepcopy

def get_initial_obs_task2():
    dataset = [{
        "sample_id": "2",
        "text": "I absolutely hate this, worst purchase ever",
        "label": "positive",
        "issues": ["wrong_label"],
        "confidence": 0.35
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

def step_task2(state, action):
    obs = deepcopy(state["obs"])
    reward = 0.0
    done = False
    
    ground_truth = "negative"
    
    if action.type == "relabel":
        new_label = action.payload.get("new_label", obs["label"])
        
        if new_label == ground_truth:
            reward += 0.5
            obs["issues"] = []
        else:
            reward -= 0.5
            
        obs["label"] = new_label
            
    elif action.type == "mark_clean":
        if obs["label"] == ground_truth:
            reward += 0.5
        else:
            reward -= 0.5
        done = True
        
    elif action.type == "remove":
        reward -= 0.3
        
    obs["step"] += 1
    obs["history"].append(f"{action.type}: {action.payload}")
    
    return obs, float(reward), done, {}

def grade_task2(state):
    final_label = state["obs"]["label"]
    ground_truth = "negative"
    
    if final_label == ground_truth:
        return 1.0
    elif state["obs"]["confidence"] < 0.4:  # initial confidence was close to threshold
        return 0.4
    else:
        return 0.0
