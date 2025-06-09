# Modal Deployment Guide

## TL;DR - Complete Example

Here's the full process from zero to deployed:

```bash
# 1. Install and setup Modal (one-time)
pip install modal
modal setup  # This opens a browser to log in

# 2. Choose a password and generate its hash
python -c "from argon2 import PasswordHasher; ph = PasswordHasher(); print(ph.hash('my-awesome-password-123'))"
# Output: $argon2id$v=19$m=65536,t=3,p=4$1234abcd...

# 3. Create the secret with the hash
modal secret create ludic-auth-secret LUDIC_PASSWORD_HASH='$argon2id$v=19$m=65536,t=3,p=4$1234abcd...'

# 4. Deploy the apps
modal deploy src/ludic_envs/trainers/modal_training_app.py
modal deploy src/ludic_envs/inference/modal_inference_app.py
# Output: ✓ Created web endpoint https://john-doe--ludic-training-grpotrainer-start-training.modal.run

# 5. Your deployment is ready! Share with friends:
#    - Modal username: john-doe (from the URL above)
#    - Password: my-awesome-password-123 (what you chose, NOT the hash)
```

## Initial Setup (One-time)

### 1. Install Modal and Authenticate
```bash
pip install modal
modal setup  # This will open a browser to log into your Modal account
```

### 2. Create the Password Secret

First, generate a password hash from your chosen password:
```bash
python -c "from argon2 import PasswordHasher; ph = PasswordHasher(); print(ph.hash('your_secret_password'))"
```

Then create the Modal secret with this hash:
```bash
modal secret create ludic-auth-secret LUDIC_PASSWORD_HASH='$argon2id$v=19$m=65536,t=3,p=4$...'
```

**Important**: The hash (starting with `$argon2id$...`) is what goes in the secret, NOT your actual password!

### 3. (Optional) Create Other Secrets

If you want to use Hugging Face models or Weights & Biases:
```bash
# For Hugging Face model access
modal secret create huggingface-secret HF_TOKEN=<your_hf_token>

# For Weights & Biases logging
modal secret create wandb-secret WANDB_API_KEY=<your_wandb_key>
```

## Deployment

### Option 1: Manual Deployment
```bash
# Deploy the training app
modal deploy src/ludic_envs/trainers/modal_training_app.py

# Deploy the inference app  
modal deploy src/ludic_envs/inference/modal_inference_app.py
```

### Option 2: Automatic Deployment via Scripts
The example scripts can deploy for you:
```bash
# This will deploy the training app AND start training
python examples/train/train_tic_tac_toe_modal.py \
    --deploy \
    --base-url <your-modal-username> \
    --password <your_actual_password>
```

## Finding Your Modal Username

After deployment, Modal will show you the endpoint URLs. They look like:
```
https://YOUR-MODAL-USERNAME--ludic-training-grpotrainer-start-training.modal.run
```

Your Modal username is the part before the first `--`. You can also find it:
1. In the Modal dashboard: https://modal.com/dashboard
2. By running: `modal profile current`

## Complete First-Time Example

```bash
# 1. Setup Modal CLI (one-time)
pip install modal
modal setup

# 2. Generate password hash
python -c "from argon2 import PasswordHasher; ph = PasswordHasher(); print(ph.hash('mysecretpass123'))"
# Output: $argon2id$v=19$m=65536,t=3,p=4$abc123...

# 3. Create secret with the hash
modal secret create ludic-auth-secret LUDIC_PASSWORD_HASH='$argon2id$v=19$m=65536,t=3,p=4$abc123...'

# 4. Deploy the apps
modal deploy src/ludic_envs/trainers/modal_training_app.py
modal deploy src/ludic_envs/inference/modal_inference_app.py

# 5. Use the deployed apps (note: using the actual password, not the hash!)
python examples/train/train_tic_tac_toe_modal.py \
    --base-url your-modal-username \
    --password mysecretpass123 \
    --model-name Qwen/Qwen2.5-7B-Instruct
```

## Sharing with Others

To let someone else use your deployed Modal apps:

1. **Give them**:
   - Your Modal username (from the deployment URLs)
   - The password you chose (NOT the hash!)
   - The example scripts

2. **They need**:
   - Python with required packages installed
   - The password you shared
   - They do NOT need a Modal account
   - They do NOT need to deploy anything

3. **They can run**:
   ```bash
   # Using your deployed infrastructure
   python -m ludic_envs.modal_client \
       --base-url your-modal-username \
       --password the-password-you-shared \
       --action train \
       --model-name Qwen/Qwen2.5-1.5B-Instruct \
       --env-type tic_tac_toe \
       --output-dir /tmp/my_model
   ```

## Security Notes

- The password hash (`$argon2id$...`) goes in Modal secrets
- The actual password is what you share with users
- Never share the hash - only share the password
- Users authenticate with the password, which is verified against the hash
- All communication is over HTTPS

## Troubleshooting

### "Password not configured" Error
Make sure you created the secret with the exact name:
```bash
modal secret list  # Check if ludic-auth-secret exists
```

### "Invalid password" Error  
- Make sure you're using the password, not the hash
- Password is case-sensitive
- Check that the secret was created correctly

### Can't Find Modal Username
After deployment, look for output like:
```
✓ Created web endpoint https://YOUR-USERNAME--ludic-training-grpotrainer-start-training.modal.run
```
Your username is `YOUR-USERNAME` in this example.