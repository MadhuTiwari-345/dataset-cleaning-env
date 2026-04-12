from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional
import json
from tasks.scoring import clamp_open_unit_interval

app = FastAPI()

class Sample(BaseModel):
    sample_id: str
    text: str
    label: str
    issues: List[str]
    confidence: float

class Observation(BaseModel):
    sample_id: str
    text: str
    label: str
    issues: List[str]
    confidence: float
    step: int
    history: List[str]
    dataset: Optional[List[Dict[str, Any]]] = None

class Action(BaseModel):
    type: str # "fix_text", "relabel", "remove", "mark_clean"
    payload: Dict[str, Any]

class Reward(BaseModel):
    value: float
    breakdown: Dict[str, float]

class ResetRequest(BaseModel):
    task_id: str = "text-cleaning"

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetResponse(BaseModel):
    observation: Observation

STATE = {
    "task_id": "text-cleaning",
    "step": 0,
    "max_steps": 5,
    "obs": None,
    "history": [],
    "done": False,
    "score": 0.0,
    "dataset": [],
    "initial_error_rate": 0.0,
    "last_action": None
}

from tasks.task1_cleaning import get_initial_obs_task1, grade_task1, step_task1
from tasks.task2_labeling import get_initial_obs_task2, grade_task2, step_task2
from tasks.task3_alignment import get_initial_obs_task3, grade_task3, step_task3

@app.post("/reset", response_model=ResetResponse)
async def reset(req: Optional[ResetRequest] = None):
    task = req.task_id if req and req.task_id else "text-cleaning"
    STATE["task_id"] = task
    STATE["step"] = 0
    STATE["history"] = []
    STATE["done"] = False
    STATE["score"] = 0.0
    STATE["last_action"] = None
    
    if task == "text-cleaning":
        STATE["max_steps"] = 5
        obs, dataset = get_initial_obs_task1()
    elif task == "label-correction":
        STATE["max_steps"] = 8
        obs, dataset = get_initial_obs_task2()
    elif task == "dataset-alignment":
        STATE["max_steps"] = 15
        obs, dataset = get_initial_obs_task3()
    else:
        obs, dataset = get_initial_obs_task1()
        
    STATE["obs"] = obs
    STATE["dataset"] = dataset
    return {"observation": obs}

@app.post("/step", response_model=StepResponse)
async def step(action: Action):
    if STATE["done"]:
        raise HTTPException(status_code=400, detail="Episode already done. Please reset.")
        
    STATE["step"] += 1
    task = STATE["task_id"]
    last_action = STATE["last_action"]
    STATE["last_action"] = action.type
    
    # Delegate step progression based on task
    if task == "text-cleaning":
        new_obs, reward, done, info = step_task1(STATE, action)
    elif task == "label-correction":
        new_obs, reward, done, info = step_task2(STATE, action)
    elif task == "dataset-alignment":
        new_obs, reward, done, info = step_task3(STATE, action)
    else:
        new_obs, reward, done, info = step_task1(STATE, action)
        
    # Check max steps
    if STATE["step"] >= STATE["max_steps"]:
        done = True
        
    if done:
        terminal_state = dict(STATE)
        terminal_state["obs"] = new_obs
        terminal_state["step"] = STATE["step"]
        terminal_state["done"] = done

        # compute final bonus
        if task == "text-cleaning":
            score = grade_task1(terminal_state)
        elif task == "label-correction":
            score = grade_task2(terminal_state)
        else:
            score = grade_task3(terminal_state)
        score = clamp_open_unit_interval(score)
        
        # Grading is added into state for logging and info
        STATE["score"] = score
        info["score"] = score
        if score > 0.8:
            reward += 1.0
            
    # Step cost
    reward -= 0.02
    
    # Repetition penalty
    if action.type == last_action and last_action is not None:
        reward -= 0.15
        
    reward = round(max(-1.0, min(1.0, reward)), 4)
    STATE["obs"] = new_obs
    STATE["done"] = done
    STATE["history"].append(f"{action.type}: {json.dumps(action.payload)}")
    
    return {"observation": new_obs, "reward": reward, "done": done, "info": info}

@app.get("/state")
async def state():
    return STATE

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
