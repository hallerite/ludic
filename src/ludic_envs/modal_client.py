"""Client for interacting with Modal apps"""

import requests
import argparse
import json
from typing import Dict, Any

class LudicModalClient:
    def __init__(self, base_url: str, password: str):
        """
        Initialize client with base URL and password
        
        Args:
            base_url: Your Modal username (e.g., "your-username")
            password: The password you set up for authentication
        """
        self.password = password
        self.training_url = f"https://{base_url}--ludic-training-grpotrainer-start-training.modal.run"
        self.eval_url = f"https://{base_url}--ludic-inference-inferenceservice-evaluate-policy.modal.run"
        self.rollout_url = f"https://{base_url}--ludic-inference-inferenceservice-generate-rollout.modal.run"
        self.health_url = f"https://{base_url}--ludic-inference-inferenceservice-health-check.modal.run"
    
    def start_training(
        self,
        model_name: str,
        env_type: str,
        output_dir: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Start a training job"""
        request_data = {
            "password": self.password,
            "model_name": model_name,
            "env_type": env_type,
            "output_dir": output_dir,
            **kwargs
        }
        
        response = requests.post(self.training_url, json=request_data)
        response.raise_for_status()
        return response.json()
    
    def evaluate_policy(
        self,
        model_path: str,
        env_type: str,
        eval_episodes: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate a trained policy"""
        request_data = {
            "password": self.password,
            "model_path": model_path,
            "env_type": env_type,
            "eval_episodes": eval_episodes,
            **kwargs
        }
        
        response = requests.post(self.eval_url, json=request_data)
        response.raise_for_status()
        return response.json()
    
    def generate_rollout(
        self,
        model_path: str,
        env_type: str,
        max_steps: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a single rollout"""
        request_data = {
            "password": self.password,
            "model_path": model_path,
            "env_type": env_type,
            "max_steps": max_steps,
            **kwargs
        }
        
        response = requests.post(self.rollout_url, json=request_data)
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health (no auth required)"""
        response = requests.get(self.health_url)
        response.raise_for_status()
        return response.json()


def main():
    parser = argparse.ArgumentParser(description="Ludic Modal Client")
    parser.add_argument("--base-url", required=True, help="Modal username (e.g., 'your-username')")
    parser.add_argument("--password", required=True, help="Authentication password")
    parser.add_argument("--action", required=True, 
                       choices=["train", "evaluate", "rollout", "health"],
                       help="Action to perform")
    
    # Training arguments
    parser.add_argument("--model-name", help="Model name for training")
    parser.add_argument("--env-type", choices=["key_door", "tic_tac_toe"], 
                       help="Environment type")
    parser.add_argument("--output-dir", help="Output directory for trained model")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    
    # Evaluation/rollout arguments
    parser.add_argument("--model-path", help="Path to trained model")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--history-strategy", default="NO_HISTORY",
                       choices=["NO_HISTORY", "FULL_HISTORY", "SCRATCHPAD"])
    
    args = parser.parse_args()
    
    # Initialize client
    client = LudicModalClient(args.base_url, args.password)
    
    try:
        if args.action == "health":
            result = client.health_check()
            print("Health check:", json.dumps(result, indent=2))
        
        elif args.action == "train":
            if not all([args.model_name, args.env_type, args.output_dir]):
                parser.error("Training requires --model-name, --env-type, and --output-dir")
            
            result = client.start_training(
                model_name=args.model_name,
                env_type=args.env_type,
                output_dir=args.output_dir,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                max_steps=args.max_steps
            )
            print("Training started:", json.dumps(result, indent=2))
        
        elif args.action == "evaluate":
            if not all([args.model_path, args.env_type]):
                parser.error("Evaluation requires --model-path and --env-type")
            
            result = client.evaluate_policy(
                model_path=args.model_path,
                env_type=args.env_type,
                eval_episodes=args.eval_episodes,
                history_strategy=args.history_strategy
            )
            print("Evaluation results:")
            print(f"Success rate: {result['results']['success_rate']:.2%}")
            print(f"Mean reward: {result['results']['mean_reward']:.3f}")
        
        elif args.action == "rollout":
            if not all([args.model_path, args.env_type]):
                parser.error("Rollout requires --model-path and --env-type")
            
            result = client.generate_rollout(
                model_path=args.model_path,
                env_type=args.env_type,
                max_steps=args.max_steps
            )
            print("Rollout summary:")
            print(f"Total reward: {result['summary']['total_reward']:.3f}")
            print(f"Steps taken: {result['summary']['steps_taken']}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")


if __name__ == "__main__":
    main()

# Example usage:
# python -m ludic_envs.modal_client --base-url your-username --password your_password --action health
# python -m ludic_envs.modal_client --base-url your-username --password your_password --action train --model-name microsoft/DialoGPT-medium --env-type key_door --output-dir /tmp/model
# python -m ludic_envs.modal_client --base-url your-username --password your_password --action evaluate --model-path /tmp/model --env-type key_door
# python -m ludic_envs.modal_client --base-url your-username --password your_password --action rollout --model-path /tmp/model --env-type tic_tac_toe