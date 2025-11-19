# Real-Time Visualization Setup Guide

## Problem
The default Qt backend may fail with errors like:
```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
```

## Solution: Use TkAgg Backend

TkAgg is a more reliable backend that usually works out of the box. Here are three ways to use it:

### Method 1: Use the Wrapper Script (Easiest)

For the full navigation demo:
```bash
python run_demo_with_gui.py -c 0.4 -i 0
```

For the minimal demo, set the backend before running:
```bash
MPLBACKEND=TkAgg python demo_pymunk_minimal_sim.py
```

### Method 2: Set Environment Variable

Before running any demo:
```bash
export MPLBACKEND=TkAgg
python demo_sim2d_ship_ice_navigation.py -c 0.4 -i 0
```

### Method 3: Modify the Demo Script

Add this at the very top of the demo script (before any imports):
```python
import os
os.environ['MPLBACKEND'] = 'TkAgg'
```

## Testing Your Setup

Test if TkAgg works:
```bash
python -c "import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pyplot as plt; fig = plt.figure(); print('TkAgg works!'); plt.close()"
```

## Available Backends

- **TkAgg** - Recommended, usually works
- **Qt5Agg** / **QtAgg** - May require Qt dependencies
- **Agg** - Non-interactive (for saving plots only)

## Troubleshooting

### If TkAgg doesn't work:

1. **Install tkinter** (usually comes with Python, but sometimes missing):
   ```bash
   # On Ubuntu/Debian:
   sudo apt-get install python3-tk
   
   # On conda:
   conda install tk
   ```

2. **Check DISPLAY variable** (if on remote server):
   ```bash
   echo $DISPLAY
   # Should show something like :0 or :10.0
   ```

3. **For remote servers**, you may need:
   - X11 forwarding: `ssh -X user@server`
   - Or use VNC/Xvfb for headless display

### If you're on a remote server without X11:

You can't use real-time visualization, but you can:
- Use `--no_anim` flag to run without GUI
- Save animation to video: set `anim.save: true` in config
- View saved plots after simulation completes

## Quick Reference

| Command | Description |
|---------|-------------|
| `MPLBACKEND=TkAgg python demo_pymunk_minimal_sim.py` | Minimal demo with GUI |
| `python run_demo_with_gui.py -c 0.4 -i 0` | Full demo with GUI |
| `python run_demo_no_gui.py -c 0.4 -i 0` | Full demo without GUI (faster) |

## Notes

- Real-time visualization is **much slower** (~10x) than running without animation
- Press **ESC** in the plot window to exit early
- The simulation will still save all plots even if you exit early

