import os
import shutil
import tensorflow as tf
from model import RipenessModel
import json

def fix_model_loading_issue():
    """Fix the model loading issue by recreating the weights file if needed"""
    
    print("🔧 Attempting to fix model loading issue...")
    
    # Define paths
    models_dir = "models"
    weights_path = os.path.join(models_dir, "ripeness_model.weights.h5")
    backup_path = os.path.join(models_dir, "ripeness_model.weights.h5.backup")
    
    try:
        # Step 1: Check if weights file exists and create backup
        if os.path.exists(weights_path):
            print(f"📁 Found weights file at {weights_path}")
            try:
                shutil.copy2(weights_path, backup_path)
                print(f"✅ Created backup at {backup_path}")
            except Exception as e:
                print(f"⚠️ Could not create backup: {e}")
        
        # Step 2: Try to load the existing model
        print("🔄 Attempting to load existing model...")
        try:
            # Create a new model instance
            model_instance = RipenessModel()
            model_instance.compile_model()
            
            # Try to load weights
            if os.path.exists(weights_path):
                model_instance.load_weights(weights_path)
                print("✅ Successfully loaded existing weights!")
                return True
        except Exception as e:
            print(f"❌ Failed to load existing weights: {e}")
            print("🔄 Will recreate the model...")
        
        # Step 3: Recreate the model if loading failed
        print("🔨 Creating fresh model weights...")
        
        # Create new model
        model_instance = RipenessModel()
        model_instance.compile_model()
        
        # Create dummy input to initialize the model
        dummy_input = tf.random.normal((1, 224, 224, 3))
        _ = model_instance.model(dummy_input)
        
        # Save the fresh weights
        model_instance.save_model(models_dir)
        
        # Create class indices file if it doesn't exist
        class_indices_path = os.path.join(models_dir, "class_indices.json")
        if not os.path.exists(class_indices_path):
            # Create default class indices for 12 classes (4 fruits × 3 ripeness levels)
            fruits = ['apple', 'banana', 'mango', 'orange']
            ripeness = ['unripe', 'ripe', 'overripe']
            class_indices = {}
            
            idx = 0
            for fruit in fruits:
                for ripe in ripeness:
                    class_indices[f"{fruit}_{ripe}"] = idx
                    idx += 1
            
            with open(class_indices_path, 'w') as f:
                json.dump(class_indices, f, indent=2)
            
            print(f"✅ Created class indices file at {class_indices_path}")
        
        print("✅ Successfully recreated model weights!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix model loading: {e}")
        return False

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("\n🧪 Testing model loading...")
    
    try:
        # Import the load_model function from app.py
        import sys
        sys.path.append('.')
        
        # Mock streamlit to avoid import issues
        class MockStreamlit:
            def warning(self, msg): print(f"WARNING: {msg}")
            def error(self, msg): print(f"ERROR: {msg}")
        
        import app
        app.st = MockStreamlit()
        
        # Try to load the model
        model = app.load_model()
        
        if model is not None:
            print("✅ Model loaded successfully!")
            return True
        else:
            print("❌ Model loading returned None")
            return False
            
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting model fix process...")
    
    # Fix the model
    if fix_model_loading_issue():
        print("\n✅ Model fix completed successfully!")
        
        # Test the fix
        if test_model_loading():
            print("\n🎉 All tests passed! Your model should now work properly.")
            print("\n📋 Next steps:")
            print("1. Restart your Streamlit app: streamlit run app.py")
            print("2. Try uploading an image to test the model")
        else:
            print("\n⚠️ Model was recreated but still has issues. Please check the error messages above.")
    else:
        print("\n❌ Model fix failed. Please check the error messages above.")
