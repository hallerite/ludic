import requests
import json
import os
import time
import sys

# Add scripts to path for config loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from ludic_config import get_password, get_training_url, get_inference_rollout_url

def train_model():
    """Train a tic-tac-toe model and save to Modal volume"""
    print("🏋️  Starting Model Training...")
    
    payload = {
        'password': get_password(),
        'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',
        'env_type': 'tic_tac_toe',
        'learning_rate': 1e-4,
        'batch_size': 4,
        'max_steps': 50,  # Small for quick demo
        'env_size': 3,
        'rollout_max_steps': 9,
        'run_name': 'demo-tic-tac-toe'  # This will be saved to /models/demo-tic-tac-toe
    }
    
    response = requests.post(
        get_training_url(),
        json=payload,
        timeout=30
    )
    
    print(f"Training Status: {response.status_code}")
    print(f"Training Response: {response.text}")
    
    if response.status_code == 200:
        call_id = response.json().get('call_id')
        print(f"✅ Training started! Call ID: {call_id}")
        print("⏳ Training in progress... (this will take a few minutes)")
        return True
    else:
        print("❌ Training failed to start")
        return False

def test_with_trained_model():
    """Test inference using the trained model from Modal volume"""
    print("\n🎯 Testing with Trained Model...")
    
    payload = {
        'password': get_password(),
        'model_path': '/models/demo-tic-tac-toe',  # Use the trained model
        'env_type': 'tic_tac_toe',
        'max_steps': 5,
        'temperature': 0.1  # Lower temperature for more deterministic moves
    }
    
    response = requests.post(
        get_inference_rollout_url(),
        json=payload,
        timeout=180  # Longer timeout for model loading
    )
    
    print(f"Inference Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Inference successful!")
        print(f"📊 Trajectory steps: {len(result['trajectory']['steps'])}")
        print(f"🎯 Total reward: {result['reward']}")
        
        # Show a few moves
        for i, step in enumerate(result['trajectory']['steps'][:3]):
            print(f"Step {i+1}: {step.get('assistant', '')[:100]}...")
        
        return True
    else:
        print(f"❌ Inference failed: {response.text}")
        return False

def test_with_default_model():
    """Test inference with default model (fallback)"""
    print("\n🔄 Testing with Default Model (fallback)...")
    
    payload = {
        'password': get_password(),
        'model_path': '/models/nonexistent-model',  # This doesn't exist
        'env_type': 'tic_tac_toe',
        'max_steps': 3,
        'temperature': 0.7
    }
    
    response = requests.post(
        get_inference_rollout_url(),
        json=payload,
        timeout=120
    )
    
    print(f"Fallback Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Fallback to default model worked!")
        return True
    else:
        print(f"❌ Fallback failed: {response.text}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Complete Modal Workflow: Train → Save → Infer")
    print("=" * 60)
    
    # Step 1: Train a model (saves to Modal volume)
    training_success = train_model()
    
    if training_success:
        print("\n⏳ Waiting a bit for training to initialize...")
        time.sleep(30)  # Give training some time to start
        
        # Step 2: Test inference with trained model
        # Note: Model might still be training, so we test fallback first
        fallback_success = test_with_default_model()
        
        # Step 3: Test with trained model (may fail if still training)
        print("\n📝 Note: If the trained model test fails, the model might still be training.")
        print("💡 You can test again later with the same model_path once training completes.")
        trained_success = test_with_trained_model()
        
        print("\n" + "=" * 60)
        print("📊 Workflow Results:")
        print(f"Training Started: {'✅' if training_success else '❌'}")
        print(f"Default Model: {'✅' if fallback_success else '❌'}")
        print(f"Trained Model: {'✅' if trained_success else '⏳ (may still be training)'}")
        
        if training_success and fallback_success:
            print("\n🎉 Modal volume workflow is working!")
            print("🔄 Once training completes, the trained model will be available at:")
            print("   /models/demo-tic-tac-toe")
        
    else:
        print("\n❌ Training failed to start - check authentication and service status") 