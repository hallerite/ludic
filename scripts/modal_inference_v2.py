import modal
from modal import App, Image, gpu, web_endpoint, Secret, Volume
from pydantic import BaseModel
from fastapi import HTTPException, Request, Depends
from typing import Optional
import os

# Fresh inference image
inference_image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install([
        "torch>=2.7.0",
        "vllm>=0.8.5.post1",
        "trl",
        "transformers",
        "fastapi",
        "pydantic",
        "git+https://github.com/hallerite/ludic.git"
    ])
)

app = App("ludic-inference-v2")
storage = Volume.from_name("ludic-models", create_if_missing=True)

class EvalRequest(BaseModel):
    password: str
    model_path: str
    env_type: str
    episodes: int = 10
    env_size: int = 4
    history_strategy: str = "NO_HISTORY"

class RolloutRequest(BaseModel):
    password: str 
    model_path: str
    env_type: str
    max_steps: int = 50
    env_size: int = 4
    temperature: float = 0.7

def verify_password(password: str) -> bool:
    expected_password = os.environ.get("LUDIC_MODAL_PASSWORD")
    if not expected_password:
        raise HTTPException(500, "Password not configured")
    print(f"🔍 Password check: received='{password}', expected='{'*' * len(expected_password)}'")
    return password == expected_password

def require_https(request: Request):
    if request.url.scheme != "https":
        raise HTTPException(400, "HTTPS required")

@app.cls(
    image=inference_image,
    gpu=gpu.A100(),
    container_idle_timeout=600,
    volumes={"/models": storage},
    secrets=[Secret.from_name("ludic-password")]
)
class InferenceServiceV2:
    
    def __init__(self):
        self.vllm_client = None
        self.rollout_gens = {}
        print("🚀 InferenceServiceV2 initialized")
    
    def _get_client(self, model_path="/models/default"):
        if self.vllm_client is None:
            from vllm import LLM
            import os
            
            print(f"🚀 Initializing vLLM model from: {model_path}")
            
            # Check if model exists in volume, otherwise use default
            if os.path.exists(model_path):
                print(f"📁 Loading trained model from volume: {model_path}")
                model_name = model_path
            else:
                print(f"📁 Model not found at {model_path}, using default: Qwen/Qwen2.5-1.5B-Instruct")
                model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            
            # Use vLLM directly instead of trying to connect to a server
            self.vllm_client = LLM(
                model=model_name, 
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8
            )
            print("✅ vLLM model initialized successfully")
        return self.vllm_client
    
    def _get_rollout_gen(self, env_type: str, max_steps: int, history: str, env_size: int):
        from ludic_envs.envs.pomdp.key_door import KeyDoorEnv
        from ludic_envs.envs.mdp.tic_tac_toe import TicTacToe
        from ludic_envs.inference.rollout_generator import RolloutGenerator, HistoryManagement
        
        key = f"{env_type}_{max_steps}_{history}_{env_size}"
        if key not in self.rollout_gens:
            env_cls = KeyDoorEnv if env_type == "key_door" else TicTacToe
            self.rollout_gens[key] = RolloutGenerator(
                env_cls=env_cls,
                max_steps=max_steps,
                history_strategy=getattr(HistoryManagement, history),
                env_kwargs={"size": env_size} if env_type == "key_door" else {}
            )
        return self.rollout_gens[key]
    
    @web_endpoint(method="POST")
    def evaluate(self, req: EvalRequest, _=Depends(require_https)):
        print(f"🧪 Evaluate endpoint hit with password: '{req.password}'")
        
        if not verify_password(req.password):
            print("❌ Password verification failed")
            raise HTTPException(401, "Invalid password")
        
        print("✅ Password verified, proceeding with evaluation")
        
        from vllm import SamplingParams
        
        client = self._get_client(req.model_path)
        rollout_gen = self._get_rollout_gen(req.env_type, 50, req.history_strategy, req.env_size)
        
        total_reward = 0
        success_count = 0
        
        for _ in range(req.episodes):
            trajectories = rollout_gen.collect(
                batch_size=1,
                model=client,
                sampling_params=SamplingParams(temperature=0.1, max_tokens=100)
            )
            
            if trajectories:
                reward = sum(step["reward"] for step in trajectories[0]["steps"])
                total_reward += reward
                if reward > 0.5:
                    success_count += 1
        
        return {
            "mean_reward": total_reward / req.episodes,
            "success_rate": success_count / req.episodes,
            "episodes": req.episodes
        }
    
    @web_endpoint(method="POST")
    def rollout(self, req: RolloutRequest, _=Depends(require_https)):
        print(f"🎲 Rollout endpoint hit with password: '{req.password}'")
        
        if not verify_password(req.password):
            print("❌ Password verification failed")
            raise HTTPException(401, "Invalid password")
        
        print("✅ Password verified, proceeding with rollout")
        
        from vllm import SamplingParams
        
        client = self._get_client(req.model_path)
        rollout_gen = self._get_rollout_gen(req.env_type, req.max_steps, "NO_HISTORY", req.env_size)
        
        trajectories = rollout_gen.collect(
            batch_size=1,
            model=client,
            sampling_params=SamplingParams(temperature=req.temperature, max_tokens=100)
        )
        
        if not trajectories:
            raise HTTPException(500, "Failed to generate trajectory")
        
        trajectory = trajectories[0]
        return {
            "trajectory": trajectory,
            "reward": sum(step["reward"] for step in trajectory["steps"]),
            "steps": len(trajectory["steps"])
        }
    
    @web_endpoint(method="GET")
    def health(self):
        print("💓 Health check")
        return {"status": "healthy", "service": "inference-v2"} 