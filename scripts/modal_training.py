import modal
from modal import App, Image, gpu, method, web_endpoint, Secret, Volume
from pydantic import BaseModel

from fastapi import HTTPException, Request, Depends
import os

# Minimal image with only required dependencies
training_image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install([
        "torch>=2.7.0",
        "vllm>=0.8.5.post1",
        "trl", 
        "transformers",
        "datasets",
        "wandb",
        "fastapi",
        "pydantic"
    ])
    .pip_install("git+https://github.com/hallerite/ludic.git")
)

app = App("ludic-training")
storage = Volume.from_name("ludic-models", create_if_missing=True)

class TrainingRequest(BaseModel):
    password: str
    model_name: str
    env_type: str  # "key_door" | "tic_tac_toe"
    learning_rate: float = 1e-5
    batch_size: int = 16
    max_steps: int = 1000
    env_size: int = 4
    rollout_max_steps: int = 50
    run_name: str = "ludic-run"

def verify_password(password: str) -> bool:
    expected_password = os.environ.get("LUDIC_MODAL_PASSWORD")
    if not expected_password:
        raise HTTPException(500, "Password not configured")
    return password == expected_password

def require_https(request: Request):
    if request.url.scheme != "https":
        raise HTTPException(400, "HTTPS required")
    return True

@app.cls(
    image=training_image,
    gpu=gpu.A100(count=2),  # 2xA100 sufficient for most models
    timeout=7200,  # 2 hours max
    volumes={"/models": storage},
    secrets=[Secret.from_name("ludic-password"), Secret.from_name("wandb-secret")]
)
class Trainer:
    
    @web_endpoint(method="POST")
    def train(self, request: TrainingRequest):
        try:
            if not verify_password(request.password):
                raise HTTPException(401, "Invalid password")
            
            call = self._run_training.spawn(request)
            return {"status": "started", "call_id": call.object_id}
        except Exception as e:
            print(f"Error in train endpoint: {e}")
            raise HTTPException(500, f"Training failed: {str(e)}")
    
    @method()
    def _run_training(self, req: TrainingRequest):
        # Set required environment variables for vLLM and PyTorch distributed training
        import os
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        
        from ludic_envs.trainers.trl.grpo import EnvGRPOTrainer
        from ludic_envs.envs.pomdp.key_door import KeyDoorEnv
        from ludic_envs.envs.mdp.tic_tac_toe import TicTacToe
        from trl import GRPOConfig
        from datasets import Dataset
        
        env_cls = KeyDoorEnv if req.env_type == "key_door" else TicTacToe
        output_dir = f"/models/{req.run_name}"
        
        # Create dummy dataset (required by GRPO but not used for env interactions)
        train_dataset = Dataset.from_dict({
            "prompt": [str(i) for i in range(req.batch_size)]
        })
        
        # Configure GRPO with required parameters
        config = GRPOConfig(
            output_dir=output_dir,
            learning_rate=req.learning_rate,
            per_device_train_batch_size=1,  # Fixed to 1 for stability
            gradient_accumulation_steps=req.batch_size,  # Use batch_size as grad accumulation
            max_steps=req.max_steps,
            num_iterations=req.max_steps,  # Required parameter
            use_vllm=True,
            vllm_mode="colocate",
            logging_steps=max(1, min(50, req.max_steps // 10)),
            save_steps=max(1, min(200, req.max_steps // 2)),
            report_to="wandb",
            run_name=req.run_name,
            bf16=True,
            beta=0.0,
            # Generation parameters (required)
            num_generations=req.batch_size,
            max_prompt_length=500 if req.env_type == "key_door" else 200,
            max_completion_length=100 if req.env_type == "key_door" else 20,
        )
        
        # Environment-specific rollout kwargs
        rollout_kwargs = {
            "group_size": req.batch_size,
            "seed": 42,
            "env_kwargs": {"size": req.env_size} if req.env_type == "key_door" else {}
        }
        
        trainer = EnvGRPOTrainer(
            env_cls=env_cls,
            model=req.model_name,
            args=config,
            train_dataset=train_dataset,
            rollout_max_steps=req.rollout_max_steps,
            rollout_kwargs=rollout_kwargs,
        )
        
        trainer.train()
        trainer.save_model(output_dir)
        
        return {"status": "completed", "model_path": output_dir} 