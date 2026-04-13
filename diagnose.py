import os
import sys
import traceback

def diagnose_model_issue():
    """Diagnose the model loading issue"""
    print("🔍 Diagnosing model loading issue...")
    print("=" * 50)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print()
    
    # Check if required modules can be imported
    modules_to_check = ['tensorflow', 'numpy', 'streamlit', 'PIL', 'model']
    
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"✅ {module} - OK")
        except ImportError as e:
            print(f"❌ {module} - FAILED: {e}")
    
    print()
    
    # Check model files
    models_dir = "models"
    if os.path.exists(models_dir):
        print("📁 Model files found:")
        for file in os.listdir(models_dir):
            file_path = os.path.join(models_dir, file)
            size = os.path.getsize(file_path)
            print(f"  - {file} ({size:,} bytes)")
    else:
        print("❌ Models directory not found")
    
    print()
    
    # Try to load the model
    try:
        print("🔄 Attempting to load model...")
        from model import RipenessModel
        
        model_instance = RipenessModel()
        print("✅ Model architecture created successfully")
        
        # Try to compile
        model_instance.compile_model()
        print("✅ Model compiled successfully")
        
        # Check if weights file exists and try to load
        weights_path = os.path.join(models_dir, "ripeness_model.weights.h5")
        if os.path.exists(weights_path):
            try:
                model_instance.load_weights(weights_path)
                print("✅ Model weights loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load weights: {e}")
                print(f"   Error type: {type(e).__name__}")
                traceback.print_exc()
        else:
            print("⚠️ No weights file found")
            
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        traceback.print_exc()
    
    print()
    print("=" * 50)
    print("🔍 Diagnosis complete!")

if __name__ == "__main__":
    diagnose_model_issue()
