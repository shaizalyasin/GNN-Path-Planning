import torch
from utils import plot_map

data = torch.load('data/dataset_v1.pt', weights_only=False)

print(f"✅ Loaded {len(data)} samples.")

# Get the first sample
sample = data[0]
print("Start:", sample['start'])
print("Goal:", sample['goal'])

# Visualize it to make sure it looks like a map
plot_map(sample['grid'], sample['path'], sample['start'], sample['goal'], title="Checking Sample #0")