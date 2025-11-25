# Webots Speedup Instructions

## To run MUCH faster:

### Option 1: Increase Simulation Speed (In Webots GUI)
1. In Webots, click the **Speed** button (top toolbar)
2. Drag the slider to **Max** (or type a large number like 10x or 100x)
3. This will make the simulation run faster than real-time

### Option 2: Run Headless (FASTEST)
1. Stop the current simulation
2. Run this command instead:
```bash
webots --mode=fast --minimize --batch /path/to/epuck_foraging.wbt
```

### Option 3: Multi-Environment Training (Advanced)
Use Stable-Baselines3's `SubprocVecEnv` to run multiple Webots instances in parallel.
(This requires more setup but can give near-linear speedup with CPU cores)

## Current Status:
- **GPU**: Neural network training (PPO updates)
- **CPU**: Webots simulation (single-threaded bottleneck)
  
**The simulation is the bottleneck, not the GPU!**
