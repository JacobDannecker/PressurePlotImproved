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

data = [
                        -544.748169,
                        -2236.290039,
                        -2460.624512,
                        -2423.094482,
                        -2175.291016,
                        -1864.910034,
                        -967.708191,
                        -331.208191,
                        -237.724548,
                        -185.575455,
                        786.495483,
                        550.785461,
                        362.640900,
                        269.612732,
                        181.036362,
                        53.290001,
                    ]

def sendData(data):
    send_data = [0, 0, 0, 0]
    [send_data[i] = d + random.random()*100 for i, d in enumerate(data)]
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
