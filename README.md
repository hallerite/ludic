# Ludic Environments
## An LLM Environments framework for the Era of Experience

## Modal Apps

Get started with password-protected Modal apps for training and inference:

### Quick Setup
1. Set up your `.env` file with your password:
   ```bash
   echo "LUDIC_MODAL_PASSWORD=your_secure_password" > .env
   ```

2. Deploy the training app:
   ```bash
   modal deploy scripts/modal_training.py
   ```

3. Deploy the inference app:
   ```bash
   modal deploy scripts/modal_inference_v2.py
   ```

### Usage
- **Training**: Send POST requests to your training endpoint with environment and model parameters
- **Inference**: Send POST requests to your inference endpoint for model rollouts
- **Authentication**: Include your password in the `Authorization` header as `Bearer your_password`

### Documentation
- [Complete Setup Guide](docs/modal_setup.md) - Detailed setup instructions
- [Development Notes](docs/modal_development_notes.md) - Technical details and troubleshooting
- [Deployment Guide](docs/modal_deployment_guide.md) - Production deployment guide

### Examples
Check out `examples/` for working code examples including `test_full_workflow.py` which demonstrates the complete train → save → infer workflow.
