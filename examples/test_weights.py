import requests
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from ludic_config import get_password, get_inference_rollout_url

# Test if your trained model from the recent training is accessible
def test_trained_model():
    # Your trained model should be at /models/tic-tac-toe-v1
    model_path = "/models/tic-tac-toe-v1"
    
    # Direct API call to test rollout with trained weights
    url = get_inference_rollout_url()
    
    data = {
        "password": get_password(),
        "model_path": model_path,  # Use your trained model
        "env_type": "tic_tac_toe",
        "temperature": 0.7,
        "max_steps": 10
    }
    
    print(f"🧪 Testing trained model: {model_path}")
    print(f"📡 Endpoint: {url}")
    
    try:
        response = requests.post(url, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Your trained model is accessible!")
            print(f"   🎯 Reward: {result['reward']}")
            print(f"   📊 Steps taken: {result['steps']}")
            print(f"   🎲 Trajectory: {len(result['trajectory']['steps'])} steps")
            return True
        else:
            print(f"❌ HTTP {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    test_trained_model() 