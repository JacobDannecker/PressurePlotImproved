"""
This file is part of PressurePlotImproved (PIMP).
PressurePlotImproved is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
PressurePlotImproved is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Foobar. If not, see <https://www.gnu.org/licenses/>. 



Contact: Jacob Dannecker 1911196@stud.hs-mannheim.de (email valid until Feb 2026).
Additional for Lift calculation in worker.py by by: David Schell 
"""

""" main.py """

from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6 import uic
from PyQt6.QtCore import QThread, pyqtSignal

from pathlib import Path

import numpy as np

import windows
import workers
from profiledata import ProfileData

import sys
import logging
import ast

# Configure Error-Logging
logging.basicConfig(
    filename="log.txt",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def logException(exc_type, exc_value, exc_traceback):
    logging.error(
        "+++++++++++++++++++++++++ Uncaught exception +++++++++++++++++++++++++",
        exc_info=(exc_type, exc_value, exc_traceback),
    )
    raise exc_value


sys.excepthook = logException


class MainWindow(QMainWindow):
    """
    Main function, creates window objects and implements multithreading.
    Connects signals and slots.

    Instance Attributes:
    self.setup_window (window.SetUpWindow): The setup window.
    self.about_window (window.AboutWindow): The about window.
    self.data_worker (workers.Data): Class for reading data.
    self.plot_worker (workers.Plot): Class for plotting.
    self.calculations_worker (workers.Calculations): Class for slow calculations.
    self.data_thread (QThread): Thread for reading data.
    self.plot_thread (QThread): Thread for plotting.
    self.calculations_thread (QThread): Thread for slow calculations.

    Signals:
    show_plot_sig: Starts the plot.

    Slots:
    switch_plot_button (bool): Set state of show_plot_button.

    Methods:
    switchPlotButton: Switch state of show_plot_button.
    closeEvent: Executed, when MainWindow object is closed.
    """

    show_plot_sig = pyqtSignal()

    def __init__(self, DEFAULTS):
        super().__init__()
        """ Initialize threads, workers and set up conncetins between
            signals and slots.
        """

        uic.loadUi("./Gui_Files/MainWindow.ui", self)

        # Create window objects
        self.setup_window = windows.SetUpWindow(DEFAULTS)
        self.about_window = windows.AboutWindow(DEFAULTS["GUI_DIR"])

        # Workers
        self.data_worker = workers.Data(DEFAULTS)
        self.plot_worker = workers.Plot(DEFAULTS)
        self.calculations_worker = workers.Calculations(DEFAULTS)

        # Create window objects
        self.setup_window = windows.SetUpWindow(DEFAULTS)

        # Threads
        self.plot_thread = QThread()
        self.data_thread = QThread()
        self.calculations_thread = QThread()

        # Signals from MainWindow
        self.show_plot_sig.connect(self.plot_worker.plot)
        self.setup_window.update_sig.connect(self.plot_worker.updateSetupData)
        self.setup_window.update_sig.connect(self.data_worker.updateSetupData)
        self.setup_window.update_sig.connect(self.calculations_worker.updateSetupData)
        self.setup_window.path_name_sig.connect(self.data_worker.updatePath)
        self.setup_window.send_serial_connections_sig.connect(
            self.data_worker.updateSerialConnections
        )
        self.setup_window.switch_plot_button_sig.connect(self.switchPlotButton)

        # Signals from Data
        self.data_worker.send_data_sig.connect(self.plot_worker.updateData)
        self.data_worker.send_data_sig.connect(self.calculations_worker.updateData)
        self.data_worker.calc_lift_splines_sig.connect(
            self.calculations_worker.calculateLiftAndSplines
        )

        self.data_worker.update_info_label_sig.connect(self.plot_worker.updateInfoLabel)
        self.data_worker.update_demo_label_sig.connect(self.plot_worker.updateDemoLabel)
        self.data_worker.switch_plot_button_sig.connect(self.switchPlotButton)
        self.data_worker.switch_save_button_sig.connect(
            self.plot_worker.plot_window.switchSaveButtonStatus
        )

        # Signal from Plot
        self.plot_worker.start_data_sig.connect(self.data_worker.readDataLoop)
        self.plot_worker.plot_window.stop_plot_sig.connect(
            self.data_worker.stopDataLoop
        )
        self.plot_worker.plot_window.send_checkbox_tuple_sig.connect(
            self.plot_worker.updateCheckboxTuple
        )
        self.plot_worker.deactivate_checkboxes_sig.connect(
            self.plot_worker.plot_window.deactivateCheckboxes
        )
        self.plot_worker.activate_checkboxes_sig.connect(
            self.plot_worker.plot_window.activateCheckboxes
        )
        self.plot_worker.plot_window.save_data_sig.connect(self.data_worker.saveData)
        self.plot_worker.plot_window.send_spinbox_vals_sig.connect(
            self.calculations_worker.updateSpinboxValues
        )
        self.plot_worker.plot_window.send_spinbox_vals_sig.connect(
            self.data_worker.updateSpinboxValues
        )
        self.plot_worker.plot_window.send_spinbox_vals_sig.connect(
            self.plot_worker.updateAngleVelocity
        )

        self.plot_worker.switch_save_button_sig.connect(
            self.plot_worker.plot_window.switchSaveButtonStatus
        )

        # Signals from Calculations
        self.calculations_worker.send_lift_splines_sig.connect(
            self.data_worker.updateLift
        )
        self.calculations_worker.send_lift_splines_sig.connect(
            self.plot_worker.updateLiftAndSplines
        )

        # Setting up the menu_bar
        self.setup_action.triggered.connect(self.setup_window.show)
        self.about_action.triggered.connect(self.about_window.show)

        # Setting up the Buttons in main_window
        self.show_plot_button.clicked.connect(self.show_plot_sig.emit)

        #  Move workers to threads
        self.plot_worker.moveToThread(self.plot_thread)
        self.data_worker.moveToThread(self.data_thread)
        self.calculations_worker.moveToThread(self.calculations_thread)

        # Start threads
        self.data_thread.start()
        self.calculations_thread.start()
        self.plot_thread.start()

    def switchPlotButton(self, status):
        """Sets state of show_plot_button (Enabled/Disabled)
        Attributes:
        status (bool): To be state of button.
        """
        self.show_plot_button.setEnabled(status)

    def closeEvent(self, event):
        """Called when main application is closed."""
        self.plot_thread.quit()
        self.plot_thread.wait()
        self.data_thread.quit()
        self.data_thread.wait()
        self.calculations_thread.quit()
        self.calculations_thread.wait()


if __name__ == "__main__":

    # Read default Values
    with open("DEFAULTS.txt", "r") as file:
        defaults_str = file.read()

    DEFAULTS = ast.literal_eval(defaults_str)

    # Read Wing Data
    filename_profile = DEFAULTS["PROFILE_DAT"]
    profile_data = ProfileData.readProfileData(filename_profile)

    # Add Wing Data to Defaults
    DEFAULTS["PROFILE_NAME"] = profile_data[0]
    DEFAULTS["X_WING_TOP"] = np.array(profile_data[1])
    DEFAULTS["X_WING_BOTTOM"] = np.array(profile_data[2])
    DEFAULTS["Y_WING_TOP"] = np.array(profile_data[3])
    DEFAULTS["Y_WING_BOTTOM"] = np.array(profile_data[4])
    data_error_status = profile_data[5]
    # Convert certain lists from DEFAULTS dict into numpy array
    convert_keys = DEFAULTS["CONVERT_KEYS"]
    for key in convert_keys:
        DEFAULTS[key] = np.array(DEFAULTS[key])

    # Create a  QApplication and the main_window
    app = QApplication([])

    # Error when profile data does not meet expectations
    if data_error_status:
        warning_title = "Failure importing profile data"
        warning_info = "Application will close"
        warning_text = "X-coordinates of the profile are not in increasing sequence. \
                Please adjust the .d file manually or use another profile. \
                For further information, see: http://airfoiltools.com/airfoil/index \
                or the user manual."

        windows.WarningWindow.showWarning(warning_title, warning_text, warning_info)
    else:
        main_window = MainWindow(DEFAULTS)
        main_window.show()
        sys.exit(app.exec())
