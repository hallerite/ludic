"""Modal app for scalable inference with password authentication"""

import modal
from modal import Image, App, gpu, web_endpoint, method, Secret
import os
from argon2 import PasswordHasher
from fastapi import HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Create inference image
inference_image = (
    Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.7.0",
        "vllm>=0.8.5.post1",
        "fastapi",
        "uvicorn",
        "pydantic",
        "argon2-cffi"
    ])
    .run_commands("cd /tmp && git clone https://github.com/hallerite/ludic.git && cd ludic && pip install -e .")
)

app = App("ludic-inference")

# Authentication models
security = HTTPBearer()

class EvaluationRequest(BaseModel):
    password: str
    model_path: str
    env_type: str
    env_config: dict = {}
    eval_episodes: int = 10
    history_strategy: str = "NO_HISTORY"

class RolloutRequest(BaseModel):
    password: str
    model_path: str
    env_type: str
    env_config: dict = {}
    max_steps: int = 50
    sampling_params: Optional[dict] = None

ph = PasswordHasher()

def require_https(request: Request):
    # Modal automatically handles HTTPS, so we'll check the forwarded headers
    forwarded_proto = request.headers.get("x-forwarded-proto")
    if forwarded_proto and forwarded_proto != "https":
        raise HTTPException(status_code=400, detail="HTTPS required")
    return True

def verify_password(provided_password: str) -> bool:
    """Verify password against stored Argon2 hash"""
    expected_hash = os.environ.get("LUDIC_PASSWORD_HASH")
    if not expected_hash:
        raise HTTPException(status_code=500, detail="Password not configured")
    try:
        ph.verify(expected_hash, provided_password)
        return True
    except Exception:
        return False

@app.cls(
    image=inference_image,
    gpu=gpu.A100(count=2),
    container_idle_timeout=300,
    allow_concurrent_inputs=10,
    secrets=[Secret.from_name("ludic-auth-secret")]  # Contains LUDIC_PASSWORD_HASH
)
class InferenceService:
    def __init__(self):
        # Initialize vLLM server and rollout generator
        self.client = None  # Initialize lazily
        self.rollout_generators = {}  # Cache generators by env type
    
    def _get_client(self):
        """Lazy initialization of vLLM client"""
        if self.client is None:
            from ludic_envs.inference.vllm_client import VLLMClient
            self.client = VLLMClient()
        return self.client
    
    def _get_rollout_generator(self, env_type: str, env_config: dict, max_steps: int, history_strategy: str):
        """Get or create rollout generator for environment"""
        from ludic_envs.envs.pomdp.key_door import KeyDoorEnv
        from ludic_envs.envs.mdp.tic_tac_toe import TicTacToeEnv
        from ludic_envs.inference.rollout_generator import RolloutGenerator, HistoryManagement
        
        env_cls = KeyDoorEnv if env_type == "key_door" else TicTacToeEnv
        history_enum = getattr(HistoryManagement, history_strategy)
        
        key = f"{env_type}_{max_steps}_{history_strategy}_{hash(str(env_config))}"
        
        if key not in self.rollout_generators:
            self.rollout_generators[key] = RolloutGenerator(
                env_cls=env_cls,
                max_steps=max_steps,
                history_strategy=history_enum,
                env_kwargs=env_config
            )
        
        return self.rollout_generators[key]
    
    @web_endpoint(method="POST", docs=True)
    def evaluate_policy(self, request: EvaluationRequest, _=Depends(require_https)):
        """Run policy evaluation with password authentication"""
        # Verify password
        if not verify_password(request.password):
            raise HTTPException(status_code=401, detail="Invalid password")
        
        try:
            client = self._get_client()
            rollout_gen = self._get_rollout_generator(
                request.env_type, 
                request.env_config, 
                50,  # max_steps for evaluation
                request.history_strategy
            )
            
            # Run evaluation episodes
            all_trajectories = []
            total_reward = 0
            success_count = 0
            
            from vllm import SamplingParams
            sampling_params = SamplingParams(temperature=0.1, max_tokens=100)
            
            for episode in range(request.eval_episodes):
                trajectories = rollout_gen.collect(
                    batch_size=1, 
                    model=client, 
                    sampling_params=sampling_params
                )
                
                if trajectories:
                    traj = trajectories[0]
                    episode_reward = sum(step["reward"] for step in traj["steps"])
                    total_reward += episode_reward
                    
                    # Count success (assuming reward > 0.5 indicates success)
                    if episode_reward > 0.5:
                        success_count += 1
                    
                    all_trajectories.append({
                        "episode": episode,
                        "reward": episode_reward,
                        "steps": len(traj["steps"]),
                        "trajectory": traj
                    })
            
            return {
                "model_path": request.model_path,
                "env_type": request.env_type,
                "eval_episodes": request.eval_episodes,
                "history_strategy": request.history_strategy,
                "results": {
                    "mean_reward": total_reward / request.eval_episodes,
                    "success_rate": success_count / request.eval_episodes,
                    "total_episodes": request.eval_episodes
                },
                "trajectories": all_trajectories
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
    
    @web_endpoint(method="POST", docs=True) 
    def generate_rollout(self, request: RolloutRequest, _=Depends(require_https)):
        """Generate single rollout trajectory with password authentication"""
        # Verify password
        if not verify_password(request.password):
            raise HTTPException(status_code=401, detail="Invalid password")
        
        try:
            client = self._get_client()
            rollout_gen = self._get_rollout_generator(
                request.env_type,
                request.env_config, 
                request.max_steps,
                "NO_HISTORY"  # Default for single rollouts
            )
            
            # Set up sampling parameters
            from vllm import SamplingParams
            if request.sampling_params:
                sampling_params = SamplingParams(**request.sampling_params)
            else:
                sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
            
            # Generate trajectory
            trajectories = rollout_gen.collect(
                batch_size=1,
                model=client,
                sampling_params=sampling_params
            )
            
            if not trajectories:
                raise HTTPException(status_code=500, detail="Failed to generate trajectory")
            
            trajectory = trajectories[0]
            episode_reward = sum(step["reward"] for step in trajectory["steps"])
            
            return {
                "model_path": request.model_path,
                "env_type": request.env_type,
                "max_steps": request.max_steps,
                "trajectory": trajectory,
                "summary": {
                    "total_reward": episode_reward,
                    "steps_taken": len(trajectory["steps"]),
                    "final_reward": trajectory["steps"][-1]["reward"] if trajectory["steps"] else 0
                }
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Rollout generation failed: {str(e)}")
    
    @web_endpoint(method="GET", docs=True)
    def health_check(self):
        """Health check endpoint (no authentication required)"""
        return {
            "status": "healthy",
            "service": "ludic-inference",
            "gpu_available": True  # Could check actual GPU status
        }