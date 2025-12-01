# Comparative Analysis: GNN-Based Path Planning vs. Classical A*

This project compares a machine learning approach (Graph Neural Networks) against a classical algorithm (A*) for path planning in 2D environments.

We are training a GNN to look at a grid map and predict the best path from point A to point B, then comparing its performance to the standard A* algorithm.

## The Goal
We want to answer: **Can a GNN learn to act like a path planner just by looking at examples?**

We will measure:
1. **Accuracy:** Does the GNN find a path? Is it close to the optimal A* path?
2. **Speed:** Is the GNN faster at inference than running the full A* search?

## How it Works
1. **Data Generation:** We generate thousands of random grid maps with obstacles.
2. **Ground Truth:** We solve these maps using A* to get the "correct" paths.
3. **Training:** We treat the grid as a graph (nodes & edges) and train a GNN to predict which edges belong to the optimal path.
4. **Testing:** We give the model new, unseen maps and reconstruct paths based on its predictions.

## Setup & Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/shaizalyasin/GNN-Path-Planning.git
   cd GNN-Path-Planning
   ```
   
2. Install the necessary libraries:
   ```bash
     pip install -r requirements.txt
   ```
## How to Run
We use main.py as the command center.

**1. Run a quick demo:**

```bash
python main.py --mode demo
```
**2. Generate Training Data:** This creates 1,000 maps and saves them to data/dataset_v1.pt.

```bash
python main.py --mode generate --samples 1000
```

## Project Structure
```
├── data/               # Stores generated datasets (.pt files)
├── models/             # Stores trained model weights (.pth files)
├── src/
│   ├── astar.py        # The A* algorithm logic
│   ├── generator.py    # Generates maps & creates datasets
│   ├── gnn_model.py    # The GNN architecture (Brain)
│   ├── train.py        # Training loop
│   └── utils.py        # Graph conversion & visualization tools
├── main.py             # Entry point for running demos & generation
└── requirements.txt    # Dependencies
```

## For the Team

**Main Branch:** Keep this clean. Only push working code here.

**New Features:** Create a new branch (e.g., feature-model-design) before you start coding.
   
