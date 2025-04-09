#!/bin/bash

# Make sure the checkpoints folder exists
mkdir -p checkpoints

# Train ResNet-20 (n=3)
echo "Training PlainNet-20..."
python3 runpb.py 3 train

# Train ResNet-56 (n=9)
echo "Training PlainNet-56..."
python3 runpb.py 9 train

# Train ResNet-110 (n=18)
echo "Training PlainNet-110..."
python3 runpb.py 18 train

echo "All PlainNet models trained and saved to checkpoints/ folder."
