# Neural MPC for Ship Ice Navigation

This directory contains a neural model predictive control (MPC) system for ship path planning. The system learns to map discrete A* graph plans to continuous paths using ship dynamics, and supports gradient-based optimization at runtime.

## Overview

The neural MPC system consists of four main components:

1. **Dataset Generation** (`generate_neural_mpc_dataset.py`): Generates training data by creating random scenarios, running A* search, and simulating ship dynamics.

2. **Training** (`train_neural_mpc.py`): Trains a neural network to predict continuous paths from discrete A* plans.

3. **Neural MPC Controller** (`ship_ice_planner/controller/neural_mpc.py`): Runtime controller that uses the trained model and supports gradient-based optimization.

4. **Demo** (`demo_dynamic_neural_mpc_v2.py`): Complete demo showing the neural MPC in action.

## Quick Start

### 1. Generate Training Dataset

```bash
python generate_neural_mpc_dataset.py --num_samples 1000 --output neural_mpc_dataset.pkl
```

This will:
- Generate 1000 random scenarios with obstacles
- Run A* search to get discrete plans
- Simulate ship tracking to get continuous paths
- Save everything to `neural_mpc_dataset.pkl`

**Note**: This may take a while (several hours for 1000 samples). You can reduce `--num_samples` for faster testing.

### 2. Train Neural Network

```bash
python train_neural_mpc.py --dataset neural_mpc_dataset.pkl --epochs 100 --batch_size 32
```

This will:
- Load the dataset
- Train a neural network to map discrete plans to continuous paths
- Save the best model to `neural_mpc_models/best_model.pth`

**Training parameters**:
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--max_discrete`: Maximum discrete nodes (default: 50)
- `--max_continuous`: Maximum continuous points (default: 500)

### 3. Run Demo

```bash
python demo_dynamic_neural_mpc_v2.py
```

This will:
- Load the trained model
- Get a discrete A* plan
- Use neural MPC to predict/optimize a continuous path
- Simulate ship tracking the optimized path
- Display comparison plots

## Architecture

### Neural Network Architecture

The neural network uses:
- **Encoder**: Bidirectional LSTM to encode the sequence of discrete waypoints
- **Context Encoder**: MLP to encode start pose, velocities, and goal
- **Decoder**: LSTM decoder to generate continuous path points
- **Output**: Projects decoder hidden states to (x, y, psi) coordinates

### Training Data Format

Each training sample contains:
- `discrete_plan`: (N, 3) discrete waypoints from A* [x, y, psi]
- `continuous_path`: (M, 3) continuous path from A* [x, y, psi]
- `executed_states`: (K, 6) actual ship states during tracking [x, y, psi, u, v, r]
- `executed_controls`: (K, 3) control inputs
- `start_pose`: (3,) initial pose
- `start_nu`: (3,) initial velocities
- `goal`: (2,) goal position
- `obstacles`: List of obstacle vertices
- `costmap`: Costmap for reference

### Runtime Optimization

At runtime, the neural MPC can:
1. **Neural Prediction**: Use the trained model to predict a continuous path from a discrete plan
2. **Gradient Optimization**: Further optimize the path using gradient descent with the costmap

The gradient optimization:
- Starts from the neural network's prediction
- Computes costmap collision costs using bilinear interpolation (differentiable)
- Adds smoothness penalties
- Updates the path using gradient descent

## Configuration

### Dataset Generation

Edit `generate_neural_mpc_dataset.py` to adjust:
- `NUM_SAMPLES`: Number of training samples
- `MIN_GOAL_DISTANCE`, `MAX_GOAL_DISTANCE`: Goal distance range
- `MIN_OBSTACLES`, `MAX_OBSTACLES`: Number of obstacles
- `OBSTACLE_MIN_R`, `OBSTACLE_MAX_R`: Obstacle size range

### Neural MPC Controller

Edit `demo_dynamic_neural_mpc_v2.py` to adjust:
- `NEURAL_MPC_MODEL_PATH`: Path to trained model
- `USE_GRADIENT_OPTIMIZATION`: Enable/disable gradient optimization
- `OPTIMIZATION_STEPS`: Number of gradient steps
- `LEARNING_RATE`: Learning rate for optimization

## Key Features

1. **Ship Dynamics Integration**: The dataset generation uses the same ship dynamics (`SimShipDynamics`) that will be used at runtime, ensuring the neural network learns realistic paths.

2. **Discrete-to-Continuous Mapping**: The network learns to smooth and optimize discrete A* plans into continuous, trackable paths.

3. **Runtime Optimization**: Gradient-based optimization allows fine-tuning paths based on the current costmap.

4. **Fast Inference**: Once trained, the neural network provides fast path predictions (much faster than diffusion-based methods).

## Comparison with Diffusion Planner

| Feature | Diffusion Planner | Neural MPC |
|---------|------------------|------------|
| Speed | Slow (many diffusion steps) | Fast (single forward pass) |
| Optimization | Iterative diffusion process | Gradient descent on costmap |
| Training | No training needed | Requires dataset + training |
| Flexibility | Highly flexible | Learned from data |
| Runtime Optimization | Built-in | Optional gradient optimization |

## Troubleshooting

### Model Not Found
If you get "Model not found" error:
1. Make sure you've trained the model first
2. Check that `NEURAL_MPC_MODEL_PATH` in the demo points to the correct file

### Out of Memory
If you run out of memory during training:
- Reduce `--batch_size`
- Reduce `--max_discrete` or `--max_continuous`
- Use a smaller dataset (`--num_samples`)

### Poor Path Quality
If the neural MPC produces poor paths:
- Train for more epochs
- Generate more training data
- Adjust the network architecture (hidden_dim, num_layers)
- Enable gradient optimization at runtime

## Future Improvements

1. **Attention Mechanism**: Add attention to better focus on relevant discrete waypoints
2. **Graph Neural Networks**: Use GNNs to better encode the discrete graph structure
3. **Online Learning**: Fine-tune the model during execution
4. **Multi-scale Planning**: Handle different planning horizons
5. **Uncertainty Estimation**: Provide confidence estimates for predictions

