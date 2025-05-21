"""
This file is part of PressurePlotImproved (PIMP).
PressurePlotImproved is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
PressurePlotImproved is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Foobar. If not, see <https://www.gnu.org/licenses/>. 



Contact: Jacob Dannecker 1911196@stud.hs-mannheim.de (email valid until Feb 2026).
Additional for Lift calculation in worker.py by by: David Schell 
"""

import numpy as np
import serial


class Device:
    """Create an  object for a device connected via a serial connection. Implements a method to request data
    from said device.
    """

    def __init__(self, device_port, device_baudrate, expected_shape):
        """Initialize devices."""
        self.connection_established = False
        self.device_port = device_port
        self.expected_shape = expected_shape

        try:
            self.ser_dev = serial.Serial(
                port=self.device_port,
                baudrate=device_baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                timeout=1,
            )
            # Set device to Request mode (Rate 0)
            self.ser_dev.reset_output_buffer()
            self.ser_dev.write(b"Rate 0\n")
            # Read answer
            line = self.ser_dev.readline()
            # Make sure input buffer is empty
            self.ser_dev.reset_input_buffer()
            self.connection_established = True
        except BaseException as err:
            self.connection_established = False
            raise Exception("ERROR connecting devices.")

    def getNewData(self):
        """Request new data from device and return it."""
        try:
            if self.connection_established:
                # Read Data
                self.ser_dev.reset_input_buffer()
                self.ser_dev.write(b"?\n")
                line = self.ser_dev.readline()
                # Convert data to numpy array of floats
                data_array = np.fromstring(line.decode("utf-8").strip(), sep="\t")
                # Check Data to match expected shape
                if data_array.shape[0] == self.expected_shape:
                    return data_array
                else:
                    self.ser_dev.close()
                    raise Exception("Received data does not have expected shape.")
        except:
            raise Exception("ERROR in method getNewData.")

    def close(self):
        self.ser_dev.close()


if __name__ == "__main__":
    port = input("Provide path to device: ")
    shape = int(input("Provide expected shape of array: "))
    baudrate = input("Provide baud rate (default is 19200): ")
    if not baudrate:
        baudrate = 19200
    else:
        baudrate = int(baudrate)
    print(f"Port: {port}, Shape: {shape}, Baudrate: {baudrate}")
    ser = Device(port, baudrate, shape)
    data = ser.getNewData()
    print(f"Shape: {data.shape}, Type: {type(data)}, Data: {data}")
