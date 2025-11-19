#!/usr/bin/env python
"""
Simple script to test if the AUTO-IceNav repository is set up correctly.
Run this after installing dependencies to verify everything works.
"""
import sys

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import pymunk
        print("✓ pymunk")
    except ImportError as e:
        print(f"✗ pymunk: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ matplotlib")
    except ImportError as e:
        print(f"✗ matplotlib: {e}")
        return False
    
    try:
        from ship_ice_planner.utils.sim_utils import generate_obstacles, init_pymunk_space
        print("✓ ship_ice_planner.utils.sim_utils")
    except ImportError as e:
        print(f"✗ ship_ice_planner: {e}")
        return False
    
    try:
        from ship_ice_planner.ship import FULL_SCALE_PSV_VERTICES
        print("✓ ship_ice_planner.ship")
    except ImportError as e:
        print(f"✗ ship_ice_planner.ship: {e}")
        return False
    
    return True

def test_ice_generation():
    """Test if ice floe generation works."""
    print("\nTesting ice floe generation...")
    
    try:
        from ship_ice_planner.utils.sim_utils import generate_obstacles
        
        obs_dicts, obstacles = generate_obstacles(
            num_obs=10,  # Just a few for testing
            min_r=4,
            max_r=10,
            min_x=0,
            max_x=80,
            min_y=0,
            max_y=100,
            seed=42
        )
        
        print(f"✓ Generated {len(obs_dicts)} ice floes")
        print(f"  - First floe center: {obs_dicts[0]['centre']}")
        print(f"  - First floe radius: {obs_dicts[0]['radius']:.2f} m")
        return True
    except Exception as e:
        print(f"✗ Ice generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pymunk_setup():
    """Test if Pymunk space can be created."""
    print("\nTesting Pymunk setup...")
    
    try:
        from ship_ice_planner.utils.sim_utils import init_pymunk_space
        
        space = init_pymunk_space()
        print(f"✓ Pymunk space created")
        print(f"  - Damping: {space.damping}")
        print(f"  - Iterations: {space.iterations}")
        return True
    except Exception as e:
        print(f"✗ Pymunk setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("AUTO-IceNav Setup Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        print("\n❌ Import tests failed. Please install dependencies:")
        print("   pip install -r requirements.txt")
        return
    
    # Test ice generation
    if not test_ice_generation():
        all_passed = False
    
    # Test Pymunk
    if not test_pymunk_setup():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! You're ready to run the demos.")
        print("\nTry running:")
        print("  python demo_pymunk_minimal_sim.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)

if __name__ == '__main__':
    main()

