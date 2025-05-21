"""
This file is part of PressurePlotImproved (PIMP).
PressurePlotImproved is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
PressurePlotImproved is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Foobar. If not, see <https://www.gnu.org/licenses/>. 



Contact: Jacob Dannecker 1911196@stud.hs-mannheim.de (email valid until Feb 2026).
Additional for Lift calculation in worker.py by by: David Schell 
"""
""" profiledata.py """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

class DataError(Exception):
    pass



class ProfileData:
    @staticmethod
    def readProfileData(filename):
        try:
            with open(filename, "r") as file:
                lines = file.readlines()

            # Get name out of NACA File and delete it
            first_line = lines.pop(0)
            name = first_line.split()[0]

            # Differentiate between two data formats
            if float(lines[0].split()[0]) > 1:
                # Lednicer Format
                length_top = int(float(lines.pop(0).split()[0]))
                lednicer_format = True
            else:
                # Selig Format
                lednicer_format = False

            # Read Dataframe
            csv_data = StringIO("".join(lines))
            data = pd.read_csv(csv_data, sep="\s+", header=None)

            # Get x_top, y_top, x_bottom, y_bottom depending on format
            x = np.array(pd.to_numeric(data[0][0:]))
            y = np.array(pd.to_numeric(data[1][0:]))

            if lednicer_format == True:
                x_top = np.array(x[0:length_top])
                y_top = np.array(y[0:length_top])
                x_bottom = np.array(x[length_top:])
                y_bottom = np.array(y[length_top:])
            else:
                # Selig format
                length_top = data[data[0] == 0.0].first_valid_index()
                x_top = np.array(x[0 : length_top + 1][::-1])
                y_top = np.array(y[0 : length_top + 1][::-1])
                x_bottom = np.array(x[length_top:])
                y_bottom = np.array(y[length_top:])

            for coords in [x_top, x_bottom]:
                # Check for increasing sequence
                if np.all(coords[:-1] < coords[1:]) == False:
                    raise DataError(
                        (
                            "WARNING! X-coordinates are not in increasing Sequence."
                            + " Please modify .dat file manually."
                        )
                    )
            return (name, x_top, x_bottom, y_top, y_bottom, 0)

        except DataError as err:
            print(err)
            return (name, x_top, x_bottom, y_top, y_bottom, err)


if __name__ == "__main__":
    filename = input("Please input path to profile (eg. ./Profiles/NACA22112b.dat: ")
    name, x_top, x_bottom, y_top, y_bottom, error = ProfileData.readProfileData(
        filename
    )

    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    ax.plot(x_top, y_top, marker="x", color="lime")
    ax.plot(x_bottom, y_bottom, marker="o", color="magenta")
    ax.axis("equal")
    ax.grid()

    plt.show()
