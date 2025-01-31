import serial
import struct

ser = serial.Serial('/dev/ttyACM0', 115200)

# Start pulse generator
ser.write(bytes([1]))

# Set interval to 5000 microseconds
ser.write(bytes([3]) + struct.pack('<I', 5000))

# Set pulse duration to 250 microseconds
ser.write(bytes([4]) + struct.pack('<I', 250))

# Disable trigger mode
ser.write(bytes([5]) + struct.pack('<I', 0))

# Stop pulse generator
ser.write(bytes([2]))