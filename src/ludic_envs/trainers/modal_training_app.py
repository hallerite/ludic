"""Modal app for distributed GRPO training with password authentication"""

import modal
from modal import Image, App, gpu, method, Secret, web_endpoint
import os
from argon2 import PasswordHasher
from fastapi import HTTPException, Request, Depends
from pydantic import BaseModel
from typing import Optional

# Create training image with all dependencies  
training_image = (
    Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.7.0",
        "vllm>=0.8.5.post1", 
        "trl",
        "transformers",
        "datasets",
        "wandb",
        "accelerate",
        "deepspeed",
        "fastapi",
        "pydantic",
        "argon2-cffi"
    ])
    .apt_install(["git"])
    .run_commands("cd /tmp && git clone https://github.com/joshuapurtell/ludic.git && cd ludic && pip install -e .")
)

app = App("ludic-training")

# Authentication models
class TrainingRequest(BaseModel):
    password: str
    model_name: str
    env_type: str  # "key_door" or "tic_tac_toe"
    output_dir: str
    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 32
    max_steps: int = 1000
    # Environment parameters
    env_size: int = 4
    rollout_max_steps: int = 50
    # vLLM parameters
    tensor_parallel_size: int = 2
    # Monitoring
    wandb_project: str = "ludic-training"

ph = PasswordHasher()

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

def require_https(request: Request):
    # Modal automatically handles HTTPS, so we'll check the forwarded headers
    forwarded_proto = request.headers.get("x-forwarded-proto")
    if forwarded_proto and forwarded_proto != "https":
        raise HTTPException(status_code=400, detail="HTTPS required")
    return True

@app.cls(
    image=training_image,
    gpu=gpu.A100(count=4),  # 4x A100 for training
    timeout=3600 * 4,       # 4 hour timeout
    secrets=[
        Secret.from_name("huggingface-secret"), 
        Secret.from_name("wandb-secret"),
        Secret.from_name("ludic-auth-secret")  # Contains LUDIC_PASSWORD_HASH
    ]
)
class GRPOTrainer:
    def __init__(self):
        # Initialize distributed training setup
        pass
    
    @web_endpoint(method="POST", docs=True)
    def start_training(self, request: TrainingRequest, _=Depends(require_https)):
        """Start a training job with password authentication"""
        # Verify password first
        if not verify_password(request.password):
            raise HTTPException(status_code=401, detail="Invalid password")
        
        # Start training job
        job_id = f"training_{hash(str(request.dict()))}"
        
        # Spawn training in background
        training_result = self.train.spawn(
            model_name=request.model_name,
            env_type=request.env_type,
            output_dir=request.output_dir,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            max_steps=request.max_steps,
            env_size=request.env_size,
            rollout_max_steps=request.rollout_max_steps,
            tensor_parallel_size=request.tensor_parallel_size,
            wandb_project=request.wandb_project
        )
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": f"Training job started for {request.model_name} on {request.env_type}",
            "modal_call_id": training_result.object_id
        }
    
    @method()
    def train(
        self,
        model_name: str,
        env_type: str,
        output_dir: str,
        learning_rate: float = 1e-5,
        batch_size: int = 32,
        max_steps: int = 1000,
        env_size: int = 4,
        rollout_max_steps: int = 50,
        tensor_parallel_size: int = 2,
        wandb_project: str = "ludic-training"
    ):
        # Main training logic using EnvGRPOTrainer
        print(f"Starting training: {model_name} on {env_type}")
        
        # Import training components
        from ludic_envs.trainers.trl.grpo import EnvGRPOTrainer
        from ludic_envs.envs.pomdp.key_door import KeyDoorEnv
        from ludic_envs.envs.mdp.tic_tac_toe import TicTacToeEnv
        from trl import GRPOConfig
        
        # Set up environment class
        env_cls = KeyDoorEnv if env_type == "key_door" else TicTacToeEnv
        
        # Configure training
        config = GRPOConfig(
            model_name=model_name,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            max_steps=max_steps,
            use_vllm=True,
            vllm_mode="colocate",
            tensor_parallel_size=tensor_parallel_size,
            output_dir=output_dir,
            logging_steps=10,
            save_steps=100,
            report_to="wandb",
            run_name=f"{model_name}_{env_type}_{wandb_project}"
        )
        
        # Initialize trainer
        trainer = EnvGRPOTrainer(
            env_cls=env_cls,
            model=model_name,
            args=config,
            rollout_max_steps=rollout_max_steps,
            env_kwargs={"size": env_size} if env_type == "key_door" else {}
        )
        
        # Run training
        trainer.train()
        
        # Save final model
        trainer.save_model(output_dir)
        
        return {
            "status": "completed",
            "output_dir": output_dir,
            "final_step": max_steps
        }

    @web_endpoint(method="GET", docs=True)
    def health_check(self):
        """Health check endpoint (no authentication required)"""
        return {
            "status": "healthy",
            "service": "ludic-training",
            "gpu_available": True
        }