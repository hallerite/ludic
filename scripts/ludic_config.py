import os
from pathlib import Path

def load_config():
    """Load configuration from config.env file"""
    config = {}
    
    # Try to load from top-level .env file
    config_file = Path(__file__).parent.parent / ".env"
    if config_file.exists():
        with open(config_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key] = value
    
    # Override with actual environment variables if they exist
    for key in config.keys():
        if key in os.environ:
            config[key] = os.environ[key]
    
    return config

# Load config once
CONFIG = load_config()

# Convenient accessors
def get_password():
    return CONFIG.get('LUDIC_MODAL_PASSWORD', '')

def get_workspace():
    return CONFIG.get('MODAL_WORKSPACE', 'your-workspace')

def get_training_url():
    return CONFIG.get('TRAINING_URL', f"https://{get_workspace()}--ludic-training-trainer-train.modal.run")

def get_inference_rollout_url():
    return CONFIG.get('INFERENCE_ROLLOUT_URL', f"https://{get_workspace()}--ludic-inference-v2-inferenceservicev-4d8233.modal.run")

def get_inference_evaluate_url():
    return CONFIG.get('INFERENCE_EVALUATE_URL', f"https://{get_workspace()}--ludic-inference-v2-inferenceservicev-1bb634.modal.run")

def get_inference_health_url():
    return CONFIG.get('INFERENCE_HEALTH_URL', f"https://{get_workspace()}--ludic-inference-v2-inferenceservicev-679265.modal.run") 