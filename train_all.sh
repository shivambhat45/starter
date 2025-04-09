#!/bin/bash

# Make sure the checkpoints folder exists
mkdir -p checkpoints

# Train ResNet-20 (n=3)
echo "Training ResNet-20..."
python3 run.py 3 train

# Train ResNet-56 (n=9)
echo "Training ResNet-56..."
python3 run.py 9 train

# Train ResNet-110 (n=18)
echo "Training ResNet-110..."
python3 run.py 18 train

echo "All ResNet models trained and saved to checkpoints/ folder."
