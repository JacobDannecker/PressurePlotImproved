"""
This file is part of PressurePlotImproved (PIMP).
PressurePlotImproved is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
PressurePlotImproved is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Foobar. If not, see <https://www.gnu.org/licenses/>. 



Contact: Jacob Dannecker 1911196@stud.hs-mannheim.de (email valid until Feb 2026).
Additional for Lift calculation in worker.py by by: David Shell 
"""
# Simulate density device using rasbperry pi pico
import random
import uasyncio as asyncio

data = [20.00, 40.00, 101300.00, 1.204]


def sendData(data):
    send_data = [0, 0, 0, 0]
    send_data[0] = data[0] + random.random() * 10
    send_data[1] = data[1] + random.random() * 10
    send_data[2] = data[2] + random.random() * 100
    send_data[3] = data[3] + random.random() * 1
    print(send_data)


status = False

while True:
    try:
        line = input().strip()
        if line == "Rate 0":
            print("Request mode activ, send '?'.")
            status = True
        if line == "?" and status:
            sendData(data)
        else:
            print("Unknown command.")

    except BaseException:
        print("Error")
        break
