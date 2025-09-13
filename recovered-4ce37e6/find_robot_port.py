#!/usr/bin/env python3
"""
Find your SO101 robot port on Windows
Easier alternative to lerobot.find_port
"""

import serial
import serial.tools.list_ports
import time

def list_all_ports():
    """List all available serial ports."""
    print("🔍 Scanning for serial ports...")
    ports = list(serial.tools.list_ports.comports())
    
    if not ports:
        print("❌ No serial ports found")
        return []
    
    print(f"Found {len(ports)} serial ports:")
    available_ports = []
    
    for port in ports:
        try:
            # Test if port is accessible
            ser = serial.Serial(port.device, 115200, timeout=1)
            ser.close()
            status = "✅ Available"
            available_ports.append(port.device)
        except:
            status = "❌ In use/inaccessible"
        
        print(f"  {port.device} - {port.description} - {status}")
        if hasattr(port, 'manufacturer') and port.manufacturer:
            print(f"    Manufacturer: {port.manufacturer}")
    
    return available_ports

def find_robot_port_interactive():
    """Interactive method to find robot port."""
    print("🤖 Interactive Robot Port Finder")
    print("=" * 40)
    
    # Step 1: Show all ports
    print("\n📋 Step 1: Current ports")
    available_ports = list_all_ports()
    
    if not available_ports:
        print("❌ No available ports found. Check:")
        print("  - USB cable connected")
        print("  - Robot powered on") 
        print("  - Driver installed")
        return None
    
    # Step 2: If only one port, suggest it
    if len(available_ports) == 1:
        port = available_ports[0]
        print(f"\n💡 Only one available port found: {port}")
        test_choice = input(f"Test connection to {port}? (y/n): ").lower()
        if test_choice == 'y':
            if test_robot_connection(port):
                return port
            else:
                print(f"❌ {port} doesn't seem to be the robot")
    
    # Step 3: Let user choose
    print(f"\n🎯 Multiple ports available: {available_ports}")
    print("Which port is your SO101 robot connected to?")
    
    for i, port in enumerate(available_ports, 1):
        print(f"  {i}. {port}")
    
    while True:
        try:
            choice = input(f"Enter choice (1-{len(available_ports)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(available_ports):
                port = available_ports[idx]
                if test_robot_connection(port):
                    return port
                else:
                    retry = input(f"❌ {port} test failed. Try another? (y/n): ")
                    if retry.lower() != 'y':
                        break
            else:
                print(f"Please enter number 1-{len(available_ports)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 Cancelled by user")
            return None
    
    return None

def test_robot_connection(port, baudrate=115200):
    """Test if port responds like a robot."""
    print(f"\n🧪 Testing connection to {port}...")
    
    try:
        # Try to open serial connection
        ser = serial.Serial(port, baudrate, timeout=2)
        time.sleep(0.5)  # Let connection stabilize
        
        # Try to write some data (basic command)
        # Note: This is just a connection test, not actual robot communication
        print(f"  ✅ Serial connection established")
        print(f"  📡 Port: {port}")
        print(f"  ⚡ Baud rate: {baudrate}")
        
        ser.close()
        
        # Ask user to confirm
        confirm = input(f"  Does this look like your robot port? (y/n): ").lower()
        return confirm == 'y'
        
    except serial.SerialException as e:
        print(f"  ❌ Serial connection failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False

def find_port_disconnect_method():
    """Original LeRobot method - disconnect/reconnect to detect."""
    print("\n🔌 Disconnect Method (like LeRobot find_port)")
    print("=" * 50)
    
    print("📋 Step 1: Finding ports with robot connected...")
    ports_before = [port.device for port in serial.tools.list_ports.comports()]
    print(f"Current ports: {ports_before}")
    
    print("\n🔌 Step 2: Disconnect your robot USB cable and press Enter...")
    input()
    
    time.sleep(1)
    ports_after = [port.device for port in serial.tools.list_ports.comports()]
    print(f"Ports after disconnect: {ports_after}")
    
    missing_ports = list(set(ports_before) - set(ports_after))
    
    if len(missing_ports) == 1:
        robot_port = missing_ports[0]
        print(f"🎯 Robot port detected: {robot_port}")
        print("🔌 Reconnect your USB cable now...")
        input("Press Enter when reconnected...")
        return robot_port
    elif len(missing_ports) == 0:
        print("❌ No ports disappeared. Robot may not be properly connected.")
        return None
    else:
        print(f"❌ Multiple ports disappeared: {missing_ports}")
        print("Try disconnecting only the robot cable.")
        return None

def main():
    print("🤖 SO101 Robot Port Finder")
    print("=" * 30)
    
    print("Choose detection method:")
    print("1. Interactive (test each port)")
    print("2. Disconnect method (unplug to detect)")
    print("3. Just list all ports")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        port = find_robot_port_interactive()
        if port:
            print(f"\n🎉 Robot port found: {port}")
            print(f"\nUse this in your commands:")
            print(f"--robot.port={port}")
            print(f"\nExample:")
            print(f"python -m lerobot.teleoperate --robot.type=so101_follower --robot.port={port} --teleop.type=hand_cv")
        else:
            print("❌ Could not determine robot port")
            
    elif choice == "2":
        port = find_port_disconnect_method()
        if port:
            print(f"\n🎉 Robot port found: {port}")
            print(f"Use: --robot.port={port}")
        else:
            print("❌ Could not determine robot port")
            
    elif choice == "3":
        list_all_ports()
        
    else:
        print("Invalid choice")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())