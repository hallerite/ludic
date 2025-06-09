import requests
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from ludic_config import get_password, get_training_url, get_workspace

# Training configuration
config = {
    "password": get_password(),
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "env_type": "tic_tac_toe",  # or "key_door" 
    "learning_rate": 1e-5,
    "batch_size": 16,
    "max_steps": 500,  # Increase for longer training
    "env_size": 4,
    "rollout_max_steps": 50,
    "run_name": "tic-tac-toe-v1"
}

# Training endpoint
training_url = get_training_url()

print("Starting training...")
print(f"Config: {json.dumps(config, indent=2)}")

response = requests.post(training_url, json=config)

if response.status_code == 200:
    result = response.json()
    print(f"✅ Training started!")
    print(f"Call ID: {result['call_id']}")
    print(f"Status: {result['status']}")
    print(f"\nMonitor at: https://modal.com/{get_workspace()}/apps/ludic-training")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text) 