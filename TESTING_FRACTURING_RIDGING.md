# Testing Ice Fracturing and Ridging

## Quick Test

Run the same simulations as before - the features are enabled by default:

```bash
# Full navigation demo with GUI
MPLBACKEND=TkAgg python run_demo_with_gui.py -c 0.4 -i 0

# Or without GUI (faster)
python run_demo_no_gui.py -c 0.4 -i 0 --output_dir output/test_fracturing
```

## What to Look For

### Ice Fracturing

**Visual indicators:**
- Large ice floes breaking into smaller pieces during collisions
- More small floes appearing over time
- Floes separating after breaking (with small velocities)

**In the console:**
- No specific messages, but you'll see more collisions as floes multiply

**In plots (after simulation):**
- `floe_mass_hist.pdf`: Should show more small floes than initial distribution
- `sim.pdf`: Final state should show more, smaller floes

**To force fracturing:**
- Use higher ice concentration (0.5) for more collisions
- Ship will hit floes harder → more fracturing

### Ice Ridging

**Visual indicators:**
- Ship slowing down when continuously pushing through ice
- Ship "stuck" or moving very slowly in dense ice areas
- More power/control effort needed (check control plots)

**In plots:**
- `control_vs_time.pdf`: Should show higher control inputs in ridge zones
- `state_vs_time.pdf`: Should show lower speeds when in ridges

**To test ridging:**
- Use straight-line path through dense ice (high concentration)
- Ship should slow down more over time as ridges form

## Specific Test Scenarios

### Test 1: High-Impact Fracturing
```bash
# High concentration = more collisions = more fracturing
python run_demo_no_gui.py -c 0.5 -i 0 --output_dir output/test_high_fracturing
```
**Expected:** Many floes fracture, creating lots of small pieces

### Test 2: Ridging Resistance
```bash
# Use straight planner to push through ice continuously
# Modify config to use 'straight' planner, or just let ship push forward
python run_demo_no_gui.py -c 0.4 -i 0 --output_dir output/test_ridging
```
**Expected:** Ship speed decreases when pushing through ice, control effort increases

### Test 3: Compare With/Without Features
```python
# Temporarily disable in ice_fracture_ridge.py:
FRACTURE_ENABLED = False
RIDGING_ENABLED = False
```
Run same simulation, compare:
- Number of floes (should be constant without fracturing)
- Ship speed (should be more consistent without ridging)
- Control effort (should be lower without ridging)

## Verification Methods

### Check Fracturing is Working

1. **Count floes:**
   ```python
   # Add to sim2d.py temporarily, or check final state
   print(f"Initial floes: {len(obs_dicts)}")
   print(f"Final floes: {len(polygons)}")
   ```
   Should see more floes at end if fracturing occurred.

2. **Check floe sizes:**
   - Look at `floe_mass_hist.pdf` - should show more small masses
   - Initial distribution vs final distribution

3. **Watch simulation:**
   - With GUI, you can see floes break in real-time
   - Look for floes splitting during high-speed collisions

### Check Ridging is Working

1. **Monitor ship speed:**
   - Check `state_vs_time.pdf` - speed should drop in ridges
   - Look for periods of slower movement

2. **Monitor control effort:**
   - Check `control_vs_time.pdf` - should see spikes/higher values
   - More thrust needed to overcome ridge resistance

3. **Check ridge zones:**
   ```python
   # Add to sim2d.py temporarily:
   if RIDGING_ENABLED:
       print(f"Active ridges: {len(ridge_zones)}")
       for r in ridge_zones:
           print(f"  Ridge at ({r.position.x:.1f}, {r.position.y:.1f}), thickness: {r.thickness:.2f}m")
   ```

## Debugging

### If fracturing doesn't happen:
- Lower `FRACTURE_BASE_THRESHOLD` (e.g., 1e5 instead of 1e6)
- Increase ship speed (more impulse per collision)
- Use higher ice concentration

### If ridging doesn't happen:
- Lower `RIDGE_MIN_FLOES` (e.g., 2 instead of 3)
- Increase `RIDGE_DETECTION_DISTANCE` (e.g., 100m)
- Make sure ship is pushing forward continuously

### Performance issues:
- Too many floes from fracturing? Increase `FRACTURE_MIN_SIZE`
- Ridging checks too frequent? Increase check interval (currently every 10 iterations)

## Recommended Test Sequence

1. **Baseline:** Run without features (disable both)
2. **Fracturing only:** Enable fracturing, disable ridging
3. **Ridging only:** Enable ridging, disable fracturing  
4. **Both enabled:** Full test

Compare results to see the impact of each feature!

