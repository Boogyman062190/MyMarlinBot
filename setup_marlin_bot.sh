#!/bin/bash

# Create directory structure
mkdir -p MyMarlinBot/.github/workflows
mkdir -p MyMarlinBot/config_templates
mkdir -p MyMarlinBot/data
mkdir -p MyMarlinBot/models
mkdir -p MyMarlinBot/scripts

# Initialize a Git repository
cd MyMarlinBot
git init

# Create README.md
cat <<EOL > README.md
# AI Marlin Firmware Bot

This repository contains the code and configurations for an AI-powered bot that generates Marlin firmware based on printer make and model.

## Directory Structure

- \`config_templates\`: Contains template configuration files for Marlin firmware.
- \`data\`: Contains JSON files with printer data for model training.
- \`models\`: Contains the trained AI model.
- \`scripts\`: Contains Python scripts for firmware generation and model training.
- \`.github/workflows\`: Contains GitHub Actions workflows for CI/CD automation.
EOL

# Create Configuration Templates
cat <<EOL > config_templates/Configuration.h.template
// Configuration.h template for Marlin firmware

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#define MOTHERBOARD BOARD_RAMPS_14_EFB
#define DEFAULT_AXIS_STEPS_PER_UNIT {80, 80, 400, 500}
// Add more configuration settings as needed

#endif // CONFIGURATION_H
EOL

cat <<EOL > config_templates/Configuration_adv.h.template
// Configuration_adv.h template for Marlin firmware

#ifndef CONFIGURATION_ADV_H
#define CONFIGURATION_ADV_H

// Advanced configuration settings

#endif // CONFIGURATION_ADV_H
EOL

# Create Data File
cat <<EOL > data/printer_data.json
[
    {
        "make": "Prusa",
        "model": "i3 MK3",
        "bed_size_x": 250,
        "bed_size_y": 210,
        "bed_size_z": 210,
        "extruder_count": 1,
        "board_type": "BOARD_PRUSA_I3_MK3"
    },
    {
        "make": "Creality",
        "model": "Ender 3",
        "bed_size_x": 220,
        "bed_size_y": 220,
        "bed_size_z": 250,
        "extruder_count": 1,
        "board_type": "BOARD_ENDER_3"
    }
]
EOL

# Create Model Training Script
cat <<EOL > scripts/train_model.py
import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
with open('data/printer_data.json', 'r') as file:
    data = json.load(file)

# Prepare features and labels
X = []
y = []
for item in data:
    X.append([item['bed_size_x'], item['bed_size_y'], item['bed_size_z'], item['extruder_count'], item['board_type']])
    y.append(item['config_settings'])

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
with open('models/firmware_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved to models/firmware_model.pkl")
EOL

# Create Firmware Generation Script
cat <<EOL > scripts/generate_firmware.py
import os
import shutil
import json
import pickle
import numpy as np
import sys

def create_firmware_directory(template_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(template_dir, output_dir)

def update_configuration_file(config_file, settings):
    with open(config_file, 'r') as file:
        lines = file.readlines()

    with open(config_file, 'w') as file:
        for line in lines:
            if 'DEFAULT_AXIS_STEPS_PER_UNIT' in line and 'steps_per_mm' in settings:
                line = f'#define DEFAULT_AXIS_STEPS_PER_UNIT {settings["steps_per_mm"]}\n'
            if 'MOTHERBOARD' in line and 'board' in settings:
                line = f'#define MOTHERBOARD {settings["board"]}\n'
            file.write(line)

def generate_firmware(printer_make, printer_model):
    with open('models/firmware_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Example features: [bed_size_x, bed_size_y, bed_size_z, extruder_count, board_type]
    features = [printer_make, printer_model]
    settings = model.predict([features])[0]

    template_dir = 'config_templates'
    output_dir = 'output_marlin'
    
    create_firmware_directory(template_dir, output_dir)
    
    config_file = os.path.join(output_dir, 'Marlin', 'Configuration.h')
    update_configuration_file(config_file, settings)

    os.system(f'platformio run -d {output_dir}')

if __name__ == "__main__":
    printer_make = sys.argv[1]
    printer_model = sys.argv[2]
    generate_firmware(printer_make, printer_model)
EOL

# Create GitHub Actions Workflow
cat <<EOL > .github/workflows/build_firmware.yml
name: Build Marlin Firmware

on:
  workflow_dispatch:
    inputs:
      make:
        description: 'Printer Make'
        required: true
      model:
        description: 'Printer Model'
        required: true

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'

    - name: Set up PlatformIO
      run: |
        python -c "import os; from pathlib import Path; import shutil; home = str(Path.home()); pio_core = f'{home}/.platformio'; os.makedirs(pio_core, exist_ok=True); os.environ['PLATFORMIO_CORE_DIR'] = pio_core"
        python -m pip install -U platformio

    - name: Install PlatformIO dependencies
      run: platformio update

    - name: Train AI Model
      run: |
        python scripts/train_model.py

    - name: Generate Firmware
      run: |
        python scripts/generate_firmware.py "${{ github.event.inputs.make }}" "${{ github.event.inputs.model }}"

    - name: Commit and Push Firmware
      run: |
        git add output_marlin
        git commit -m "Add generated firmware"
        git push
EOL

# Create Initial Commit
git add .
git commit -m "Initial commit with setup scripts and configurations"
git branch -M main
git remote add origin https://github.com/yourusername/MyMarlinBot.git
git push -u origin main
