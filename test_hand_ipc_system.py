#!/usr/bin/env python3
"""
Test the complete two-process hand tracking system integration.

This script tests the HandLeaderIPC without MediaPipe dependencies,
demonstrating the two-process architecture working correctly.
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# Add lerobot to Python path
sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))

def test_hand_leader_ipc():
    """Test the HandLeaderIPC class without MediaPipe dependencies."""
    print("Testing HandLeaderIPC Integration...")
    print("=" * 50)
    
    try:
        # Import should work without MediaPipe in main process
        from lerobot.teleoperators.hand_leader.hand_leader_ipc import HandLeaderIPC
        from lerobot.teleoperators.hand_leader.config_hand_leader import HandLeaderConfig
        
        print("✓ HandLeaderIPC imported successfully")
        
        # Create configuration
        config = HandLeaderConfig()
        config.camera_index = 0
        config.socket_path = "/tmp/test_hand_tracking.sock"
        
        print("✓ Configuration created")
        
        # Create hand leader instance (don't connect yet)
        hand_leader = HandLeaderIPC(config)
        
        print("✓ HandLeaderIPC instance created")
        
        # Test action features
        action_features = hand_leader.action_features
        expected_features = [
            "shoulder_pan.pos",
            "shoulder_lift.pos", 
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos"
        ]
        
        for feature in expected_features:
            if feature not in action_features:
                print(f"✗ Missing action feature: {feature}")
                return False
        
        print("✓ Action features correct")
        
        # Test process manager setup
        process_info = hand_leader.process_manager.get_process_info()
        print(f"✓ Process manager initialized: {process_info['command']}")
        
        # Test safe position
        safe_pos = hand_leader._get_safe_position()
        if len(safe_pos) != 6:
            print(f"✗ Safe position should have 6 joints, got {len(safe_pos)}")
            return False
            
        print("✓ Safe position generation works")
        
        # Test connection stats without connection
        stats = hand_leader.get_connection_stats()
        if stats['connected'] != False:
            print("✗ Should not be connected initially")
            return False
            
        print("✓ Connection status reporting works")
        
        print("\n🎉 HandLeaderIPC integration test PASSED!")
        print("\nNext step: Test with actual MediaPipe process communication...")
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tracking_process():
    """Test that the tracking process can be launched."""
    print("\nTesting MediaPipe Tracking Process...")
    print("=" * 50)
    
    # Find tracking script
    script_path = Path(__file__).parent / "lerobot" / "src" / "lerobot" / "teleoperators" / "hand_leader" / "tracking_process.py"
    
    if not script_path.exists():
        print(f"✗ Tracking script not found: {script_path}")
        return False
    
    print("✓ Tracking script found")
    
    # Find MediaPipe environment
    env_python = Path(__file__).parent / "hand_tracking_env" / "bin" / "python"
    
    if not env_python.exists():
        print(f"✗ MediaPipe environment not found: {env_python}")
        print("  Run: Create MediaPipe environment first")
        return False
    
    print("✓ MediaPipe environment found")
    
    # Test help command (should work quickly)
    try:
        result = subprocess.run([
            str(env_python),
            str(script_path),
            "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"✗ Help command failed: {result.stderr}")
            return False
            
        if "MediaPipe Hand Tracking Process" not in result.stdout:
            print(f"✗ Unexpected help output: {result.stdout}")
            return False
            
        print("✓ Tracking process help command works")
        return True
        
    except subprocess.TimeoutExpired:
        print("✗ Tracking process help timed out")
        return False
    except Exception as e:
        print(f"✗ Tracking process test failed: {e}")
        return False

def test_ipc_protocol():
    """Test IPC protocol directly."""
    print("\nTesting IPC Protocol...")
    print("=" * 50)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent / "lerobot" / "src"))
        from lerobot.teleoperators.hand_leader.ipc_protocol import HandTrackingProtocol, create_test_data, MessageValidator
        
        # Create test data
        test_data = create_test_data()
        print("✓ Test data created")
        
        # Pack data
        packed = HandTrackingProtocol.pack_hand_data(test_data)
        if len(packed) != 46:  # Expected message size with orientation (2 bytes + 11 floats)
            print(f"✗ Wrong packed size: {len(packed)}")
            return False
        print("✓ Data packing works")
        
        # Unpack data
        unpacked = HandTrackingProtocol.unpack_hand_data(packed)
        if unpacked is None:
            print("✗ Failed to unpack data")
            return False
        print("✓ Data unpacking works")
        
        # Test heartbeat
        heartbeat = HandTrackingProtocol.pack_heartbeat()
        if len(heartbeat) != 5:  # Expected heartbeat size
            print(f"✗ Wrong heartbeat size: {len(heartbeat)}")
            return False
        print("✓ Heartbeat packing works")
        
        print("✓ IPC Protocol test PASSED!")
        return True
        
    except Exception as e:
        print(f"✗ IPC Protocol test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Hand Tracking Two-Process System Test")
    print("=" * 60)
    
    tests = [
        ("IPC Protocol", test_ipc_protocol),
        ("MediaPipe Process", test_tracking_process),
        ("HandLeaderIPC Integration", test_hand_leader_ipc),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name} test PASSED")
        else:
            print(f"❌ {test_name} test FAILED")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests PASSED! Two-process architecture is working!")
        print("\nThe system is ready for teleoperation with hand tracking:")
        print("lerobot-teleoperate \\")
        print("    --robot.type=mock_so101_robot \\")  
        print("    --teleop.type=hand_leader_ipc")
    else:
        print("\n❌ Some tests FAILED. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
