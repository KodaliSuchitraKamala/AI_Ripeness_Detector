@echo off
echo 🔧 Fixing model loading issue...
echo.

REM Define paths
set MODELS_DIR=models
set WEIGHTS_PATH=%MODELS_DIR%\ripeness_model.weights.h5
set BACKUP_PATH=%MODELS_DIR%\ripeness_model.weights.h5.backup

REM Step 1: Create backup if weights file exists
if exist "%WEIGHTS_PATH%" (
    echo 📁 Found weights file, creating backup...
    copy "%WEIGHTS_PATH%" "%BACKUP_PATH%" >nul
    if exist "%BACKUP_PATH%" (
        echo ✅ Backup created successfully
    ) else (
        echo ⚠️ Could not create backup
    )
) else (
    echo 📁 No weights file found, will create new one...
)

REM Step 2: Run Python script to recreate model
echo 🔄 Recreating model weights...
python -c "
import os
import shutil
import tensorflow as tf
from model import RipenessModel
import json

print('🔨 Creating fresh model weights...')

# Create new model
model_instance = RipenessModel()
model_instance.compile_model()

# Create dummy input to initialize the model
dummy_input = tf.random.normal((1, 224, 224, 3))
_ = model_instance.model(dummy_input)

# Save the fresh weights
model_instance.save_model('models')

# Create class indices file if it doesn't exist
class_indices_path = os.path.join('models', 'class_indices.json')
if not os.path.exists(class_indices_path):
    fruits = ['apple', 'banana', 'mango', 'orange']
    ripeness = ['unripe', 'ripe', 'overripe']
    class_indices = {}
    
    idx = 0
    for fruit in fruits:
        for ripe in ripeness:
            class_indices[f'{fruit}_{ripe}'] = idx
            idx += 1
    
    with open(class_indices_path, 'w') as f:
        json.dump(class_indices, f, indent=2)
    
    print(f'✅ Created class indices file')

print('✅ Model weights recreated successfully!')
"

if %ERRORLEVEL% EQU 0 (
    echo ✅ Model fix completed successfully!
    echo.
    echo 📋 Next steps:
    echo 1. Close this window
    echo 2. Restart your Streamlit app: streamlit run app.py
    echo 3. Try uploading an image to test the model
) else (
    echo ❌ Model fix failed. Please check the error messages above.
)

pause
