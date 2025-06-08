# Modal Apps Setup Guide for Ludic Environments

This guide covers how to set up and use the Modal apps for distributed training and inference of RL models using ludic environments.

## Prerequisites

1. Install Modal CLI:
```bash
pip install modal
modal setup
```

2. Create Modal account at https://modal.com

## Authentication Setup

### 1. Generate Password Hash

Replace `your_secret_password` with your actual password:

```bash
python -c "from argon2 import PasswordHasher; ph = PasswordHasher(); print(ph.hash('your_secret_password'))"
```

Save the output hash - you'll need it for the next step.

### 2. Create Modal Secrets

Create the authentication secret:
```bash
modal secret create ludic-auth-secret LUDIC_PASSWORD_HASH=<your_hash_from_step_1>
```

Create Hugging Face secret (optional, for model downloads):
```bash
modal secret create huggingface-secret HF_TOKEN=<your_huggingface_token>
```

Create Weights & Biases secret (optional, for training monitoring):
```bash
modal secret create wandb-secret WANDB_API_KEY=<your_wandb_api_key>
```

### 3. Deploy the Apps

Deploy the training app:
```bash
modal deploy src/ludic_envs/trainers/modal_training_app.py
```

Deploy the inference app:
```bash
modal deploy src/ludic_envs/inference/modal_inference_app.py
```

After deployment, Modal will show you the endpoint URLs. Save these URLs.

## Usage Examples

### Training a Model

```python
import requests

# Start a training job
training_request = {
    "password": "your_secret_password",
    "model_name": "microsoft/DialoGPT-medium", 
    "env_type": "key_door",  # or "tic_tac_toe"
    "output_dir": "/tmp/trained_model",
    "learning_rate": 1e-5,
    "batch_size": 16,
    "max_steps": 500,
    "env_size": 4,
    "rollout_max_steps": 50,
    "wandb_project": "my_ludic_experiment"
}

response = requests.post(
    "https://your-username--ludic-training-grpotrainer-start-training.modal.run",
    json=training_request
)
print(response.json())
```

### Running Policy Evaluation

```python
import requests

# Evaluate a trained model
eval_request = {
    "password": "your_secret_password",
    "model_path": "/path/to/trained/model",
    "env_type": "key_door", 
    "env_config": {"size": 4},
    "eval_episodes": 10,
    "history_strategy": "NO_HISTORY"  # or "FULL_HISTORY", "SCRATCHPAD"
}

response = requests.post(
    "https://your-username--ludic-inference-inferenceservice-evaluate-policy.modal.run",
    json=eval_request
)
results = response.json()
print(f"Success rate: {results['results']['success_rate']}")
print(f"Mean reward: {results['results']['mean_reward']}")
```

### Generating Single Rollout

```python
import requests

# Generate a single trajectory
rollout_request = {
    "password": "your_secret_password",
    "model_path": "/path/to/model",
    "env_type": "tic_tac_toe",
    "max_steps": 20,
    "sampling_params": {"temperature": 0.8, "max_tokens": 50}
}

response = requests.post(
    "https://your-username--ludic-inference-inferenceservice-generate-rollout.modal.run", 
    json=rollout_request
)
trajectory = response.json()
print(f"Total reward: {trajectory['summary']['total_reward']}")
print(f"Steps taken: {trajectory['summary']['steps_taken']}")
```

### Health Check (No Authentication)

```python
import requests

response = requests.get(
    "https://your-username--ludic-inference-inferenceservice-health-check.modal.run"
)
print(response.json())
```

### Using the Command-Line Client

The package includes a command-line client for easier interaction:

```bash
# Health check
python -m ludic_envs.modal_client --base-url your-username --password your_password --action health

# Start training
python -m ludic_envs.modal_client --base-url your-username --password your_password \
    --action train --model-name microsoft/DialoGPT-medium --env-type key_door \
    --output-dir /tmp/model --max-steps 500

# Evaluate model
python -m ludic_envs.modal_client --base-url your-username --password your_password \
    --action evaluate --model-path /tmp/model --env-type key_door \
    --eval-episodes 20 --history-strategy FULL_HISTORY

# Generate rollout
python -m ludic_envs.modal_client --base-url your-username --password your_password \
    --action rollout --model-path /tmp/model --env-type tic_tac_toe \
    --max-steps 20
```

## API Documentation

Once deployed, you can access interactive API documentation:
- Training API: `https://your-username--ludic-training-grpotrainer-start-training.modal.run/docs`
- Inference API: `https://your-username--ludic-inference-inferenceservice-evaluate-policy.modal.run/docs`

## Environment Configuration

### KeyDoor Environment
```python
env_config = {
    "size": 4,  # Grid size (4x4)
    # Add other KeyDoor-specific parameters
}
```

### TicTacToe Environment
```python
env_config = {}  # TicTacToe uses default configuration
```

## Training Parameters

- `model_name`: Hugging Face model ID (e.g., "microsoft/DialoGPT-medium")
- `learning_rate`: Learning rate for training (default: 1e-5)
- `batch_size`: Batch size per GPU (default: 32)
- `max_steps`: Maximum training steps (default: 1000)
- `tensor_parallel_size`: Number of GPUs for tensor parallelism (default: 2)
- `wandb_project`: Weights & Biases project name

## History Strategies

- `NO_HISTORY`: No history tracking
- `FULL_HISTORY`: Complete conversation history
- `SCRATCHPAD`: Scratchpad-based history management

## Monitoring

View job status and logs in the Modal dashboard:
1. Go to https://modal.com/dashboard
2. Navigate to your apps
3. Check running jobs and logs

## Cost Control

- Set appropriate timeouts for training jobs
- Use `container_idle_timeout` for inference to auto-shutdown
- Monitor GPU usage in Modal dashboard
- Set `allow_concurrent_inputs` to limit parallel requests

## Troubleshooting

### Common Issues

1. **Authentication Failed**
   - Verify password matches the one used to generate hash
   - Check that `ludic-auth-secret` is properly created

2. **Import Errors**
   - Ensure the ludic package is properly installed in the image
   - Check that all dependencies are specified

3. **GPU Out of Memory**
   - Reduce batch size
   - Adjust tensor_parallel_size
   - Use smaller model

4. **Timeout Errors**
   - Increase timeout in app configuration
   - Check if training is actually progressing

### Debug Commands

Check secret exists:
```bash
modal secret list
```

View app logs:
```bash
modal app logs ludic-training
modal app logs ludic-inference
```

## Security Notes

- Always use HTTPS endpoints (Modal enforces this)
- Never share your password or hash
- Rotate passwords periodically
- Monitor access logs in Modal dashboard
- Use environment-specific passwords for dev/prod

## Sharing Access

To give someone access to your deployed apps:

1. Share the password (not the hash!)
2. Share the Modal endpoint URLs
3. Provide this documentation
4. Consider creating separate passwords for different users

Remember: They only need the password and URLs to use your infrastructure!