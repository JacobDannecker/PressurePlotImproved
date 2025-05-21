"""
This file is part of PressurePlotImproved (PIMP).
PressurePlotImproved is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
PressurePlotImproved is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Foobar. If not, see <https://www.gnu.org/licenses/>. 



Contact: Jacob Dannecker 1911196@stud.hs-mannheim.de (email valid until Feb 2026).
Additional for Lift calculation in worker.py by by: David Schell 
"""

""" windows.py """

from PyQt6 import uic
from PyQt6.QtWidgets import (
    QWidget,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QPushButton,
    QMessageBox,
)
from PyQt6.QtCore import pyqtSignal, pyqtSlot
from pathlib import Path

import numpy as np
from datetime import datetime

import serialdevices
import logging


class SetUpWindow(QWidget):
    """
    This class defines the appearance and behavior of the setup window.

    Instance attributes:
        pressure_line_edit (QLineEdit): Port of the pressure scanner.
        density_line_edit (QLineEdit): Port of the density scanner.
        checkbox_list (list of QCheckBox): Status of pressure channels.
        checkbox_color_list (list of QCheckBox): Color coding for pressure channels.
        spinbox_list (list of QDoubleSpinBox): Coordinates for pressure wall bores.
        path_label (QLabel): Displays the path to the file in which data will be saved.
        open_file_button (QPushButton): Button to open file dialog.
        button_box (QDialogButtonBox): Accepts or rejects changes.

    Signals:
        update_sig (tuple): Emits two masks to filter out active channels and the
                            x-coordinates of the wall bores.
        path_name_sig (tuple): Emits the path to the file in which data will be saved.
        send_serial_connections_sig (tuple): Emits serial connection objects.
        switch_plot_button_sig (bool): Emits the new status of show_plot_button.

    Methods:
        __init__:
       setUp: Set up the application and enable plotting.
       changeStateOfSpinbox: Changes status of spin boxes.
       openFileDialog: Opens file dialog.
    """

    update_sig = pyqtSignal(tuple)
    path_name_sig = pyqtSignal(tuple)
    send_serial_connections_sig = pyqtSignal(tuple)
    switch_plot_button_sig = pyqtSignal(bool)

    def __init__(self, DEFAULTS):
        super().__init__()

        # Load defaults and .ui file
        uic.loadUi(DEFAULTS["GUI_DIR"] + "SetUpWindow.ui", self)
        # Path and filename to save data
        self.path = Path.cwd() / Path(
            DEFAULTS["DATA_DIR"] + datetime.now().strftime("%y-%m-%d_%H:%M:%S.csv")
        )
        # Default variables
        self.x_coordinates_status = DEFAULTS["X_COORDS_STATUS"]
        self.x_coordinates = DEFAULTS["X_COORDS"]
        self.pressure_port = DEFAULTS["PRESSURE_PORT"]
        self.density_port = DEFAULTS["DENSITY_PORT"]
        self.baud_rate = DEFAULTS["BAUD_RATE"]
        self.color_code = DEFAULTS["COLOR_CODE"]
        self.expected_shape_pressure = DEFAULTS["SHAPE_PRESSURE"]
        self.expected_shape_density = DEFAULTS["SHAPE_DENSITY"]
        self.demo_mode = DEFAULTS["DEMO_MODE"]

        # Create lists of check boxes and spin boxes in setup window
        self.checkbox_list = self.checkbox_group.findChildren(QCheckBox)
        self.checkbox_color_list = self.color_group.findChildren(QCheckBox)
        self.spinbox_list = self.spinbox_group.findChildren(QDoubleSpinBox)

        # Set up check boxes according to defaults
        for i, checkbox in enumerate(self.checkbox_list):
            if self.x_coordinates_status[i]:
                checkbox.setChecked(True)
                self.spinbox_list[i].setEnabled(True)
            else:
                checkbox.setChecked(False)
                self.spinbox_list[i].setEnabled(False)

        for i, checkbox in enumerate(self.checkbox_color_list):
            if self.color_code[i]:
                checkbox.setChecked(True)
            else:
                checkbox.setChecked(False)
        # Set up spin boxes according to defaults
        for i, spinbox in enumerate(self.spinbox_list):
            spinbox.setValue(self.x_coordinates[i])
        # Toggle state of spin box according to checkbox
        for checkbox in self.checkbox_list:
            checkbox.toggled.connect(self.changeStateOfSpinBoxes)

        # Set Text of QLinedits
        self.pressure_line_edit.setText(self.pressure_port)
        self.density_line_edit.setText(self.density_port)

        # Setting text for path_label
        self.path_label.setText(str(self.path))

        # Connect buttons
        self.open_file_button.clicked.connect(self.openFileDialog)
        self.button_box.rejected.connect(self.close)
        self.button_box.accepted.connect(self.setUp)

    def setUp(self):
        """
        Applies all defaults and changes made by user. Tries to connect to the
        serial devices. If successful enables the show plot button in the
        main window.

        Signals:
           update_sig (tuple): Emits two masks to filter out active channels and the
                              x-coordinates of the wall bores.
           path_name_sig (tuple): Emits the path to the file in which data will be saved.
           send_serial_connections_sig (tuple): Emits serial connection objects.
           switch_plot_button_sig (bool): Emits the new status of show_plot_button.

        """
        # Get paths to devices
        self.pressure_port = self.pressure_line_edit.text()
        self.density_port = self.density_line_edit.text()

        # Makes a List  of active points
        for i, checkbox in enumerate(self.checkbox_list):
            if checkbox.isChecked():
                self.x_coordinates_status[i] = True
            else:
                self.x_coordinates_status[i] = False

        # Read x_coordinates from spin boxes
        for i, spinbox in enumerate(self.spinbox_list):
            self.x_coordinates[i] = spinbox.value()

        # Setup color code for plot
        for i, checkbox in enumerate(self.checkbox_color_list):
            if checkbox.isChecked():
                self.color_code[i] = True
            else:
                self.color_code[i] = False

        # Create masks to filter the raw data and show only
        # the points marked as active. Also make them the right color
        # according to check boxes.

        self.mask_top = np.bitwise_and(
            self.x_coordinates_status.astype(bool), self.color_code.astype(bool)
        )

        self.mask_bottom = np.bitwise_and(
            self.x_coordinates_status.astype(bool), ~self.color_code.astype(bool)
        )

        # Check for duplicates
        coordinates_top = self.x_coordinates[self.mask_top]
        _, unique_indices_top = np.unique(coordinates_top, axis=0, return_index=True)
        all_indices_top = np.arange(len(coordinates_top))
        duplicate_indices_top = np.setdiff1d(all_indices_top, unique_indices_top)

        coordinates_bottom = self.x_coordinates[self.mask_bottom]
        _, unique_indices_bottom = np.unique(
            coordinates_bottom, axis=0, return_index=True
        )
        all_indices_bottom = np.arange(len(coordinates_bottom))
        duplicate_indices_bottom = np.setdiff1d(
            all_indices_bottom, unique_indices_bottom
        )

        # Check min length of arrays for lift calculation
        if np.sum(self.mask_top) < 2 or np.sum(self.mask_bottom) < 2:
            min_length_ok = False
        else:
            min_length_ok = True

        if not min_length_ok:
            warning_title = "Warning"
            warning_text = "Top and bottom need at least two points for interpolation."
            warning_info = "Setup will abbort."
            WarningWindow.showWarning(warning_title, warning_text, warning_info)
            self.switch_plot_button_sig.emit(False)

        if duplicate_indices_top.any() or duplicate_indices_bottom.any():
            warning_title = "Warning"
            warning_text = "Please check for duplicates."
            warning_info = "Setup will abbort."
            WarningWindow.showWarning(warning_title, warning_text, warning_info)
            self.switch_plot_button_sig.emit(False)

        if (
            not duplicate_indices_top.any()
            and not duplicate_indices_bottom.any()
            and min_length_ok == True
        ):
            # Send masks to plot worker
            self.update_sig.emit((self.mask_top, self.mask_bottom, self.x_coordinates))

            # Emit path name to Data worker
            self.path_name_sig.emit((self.path,))

            # Establish serial connection to devices
            if self.demo_mode:
                self.switch_plot_button_sig.emit(True)
            else:
                try:
                    self.pressure_device = serialdevices.Device(
                        self.pressure_port, self.baud_rate, self.expected_shape_pressure
                    )
                    self.density_device = serialdevices.Device(
                        self.density_port, self.baud_rate, self.expected_shape_density
                    )
                    self.send_serial_connections_sig.emit(
                        (self.pressure_device, self.density_device)
                    )
                    self.switch_plot_button_sig.emit(True)
                    # Check devices by reading data. If not connected correctly, 
                    # raises error
                    self.pressure_device.getNewData()
                    self.density_device.getNewData()
                except BaseException as err:
                    warning_title = "Warning"
                    warning_text = 'Could not connect devices. List ports by \
                            executing "ls /dev/ttyACM*" in a terminal window. \
                            Make sure the paths are not swapped.'
                    warning_info = "Setup will abort."
                    WarningWindow.showWarning(warning_title, warning_text, warning_info)
                    logging.error(str(err), exc_info=True)
                    self.switch_plot_button_sig.emit(False)

        self.close()

    def changeStateOfSpinBoxes(self):
        """
        Enables or disables spin boxes in setup_dialog according to check boxes.
        """
        for i, checkbox in enumerate(self.checkbox_list):
            if checkbox.isChecked():
                self.spinbox_list[i].setEnabled(True)
            else:
                self.spinbox_list[i].setEnabled(False)

    def openFileDialog(self):
        """
        Opens QFileDialogg and emits a Path Object
        """
        filename, ok = QFileDialog.getSaveFileName(
            self, "Select a File", "./Data/", "Text Files (*.csv)"
        )

        if ok:
            if ".csv" in filename:
                self.path = Path(filename)
            else:
                self.path = Path(filename + ".csv")

        self.path_label.setText(str(self.path))
        self.path_name_sig.emit((self.path,))


class AboutWindow(QWidget):
    """Loads the .ui file for the about_dialog."""

    def __init__(self, GUI_FILES):
        super().__init__()
        uic.loadUi(GUI_FILES + "AboutWindow.ui", self)
        # React to close button
        self.button_box.rejected.connect(self.close)


class PlotWindow(QWidget):
    stop_plot_sig = pyqtSignal()
    calculate_lift_sig = pyqtSignal()
    save_data_sig = pyqtSignal()
    send_checkbox_tuple_sig = pyqtSignal(tuple)
    send_spinbox_vals_sig = pyqtSignal(tuple)

    def __init__(self, DEFAULTS):
        super().__init__()
        self.test = uic.loadUi(DEFAULTS["GUI_DIR"] + "PlotWindow.ui", self)
        self.angle = DEFAULTS["ANGLE"]
        self.velocity = DEFAULTS["VELOCITY"]
        self.width = DEFAULTS["WIDTH"]
        self.checkbox_tuple = DEFAULTS["CHECKBOX_TUPLE"]
        self.spinbox_values = (self.angle, self.velocity, self.width)

        self.pause_check_box.clicked.connect(self.sendCheckboxTuple)
        self.line_check_box.clicked.connect(self.sendCheckboxTuple)
        self.cp_check_box.clicked.connect(self.sendCheckboxTuple)
        self.save_data_button.clicked.connect(self.saveDataSignal)

        # Update Static LCDs
        self.angle_spin_box.valueChanged.connect(self.angle_lcd.display)
        self.velocity_spin_box.valueChanged.connect(self.velocity_lcd.display)
        self.width_spin_box.valueChanged.connect(self.width_lcd.display)

        # Change of spin box values
        self.angle_spin_box.valueChanged.connect(self.sendSpinboxValues)
        self.velocity_spin_box.valueChanged.connect(self.sendSpinboxValues)
        self.width_spin_box.valueChanged.connect(self.sendSpinboxValues)

        # Set default Values to spin boxes and according LCDs
        self.angle_lcd.display(self.angle)
        self.velocity_lcd.display(self.velocity)
        self.width_lcd.display(self.width)
        self.angle_spin_box.setValue(self.angle)
        self.velocity_spin_box.setValue(self.velocity)
        self.width_spin_box.setValue(self.width)

        # Set Check boxes according to checkbox_tuple
        self.checkbox_list = self.checkbox_frame.findChildren(QCheckBox)
        for i, checkbox in enumerate(self.checkbox_list):
            checkbox.setChecked(self.checkbox_tuple[i])

        # Enable save_data_button
        self.switchSaveButtonStatus(True)

    def sendSpinboxValues(self):
        self.spinbox_values = (
            self.angle_spin_box.value(),
            self.velocity_spin_box.value(),
            self.width_spin_box.value(),
        )
        self.send_spinbox_vals_sig.emit(self.spinbox_values)

    def sendCheckboxTuple(self):
        self.checkbox_tuple = (
            self.pause_check_box.isChecked(),
            self.line_check_box.isChecked(),
            self.cp_check_box.isChecked(),
        )
        self.send_checkbox_tuple_sig.emit(self.checkbox_tuple)

    def deactivateCheckboxes(self):
        self.line_check_box.setEnabled(False)
        self.cp_check_box.setEnabled(False)

    def activateCheckboxes(self):
        self.line_check_box.setEnabled(True)
        self.cp_check_box.setEnabled(True)

    def saveDataSignal(self):
        # Stop normal reading of data in data_worker
        self.stop_plot_sig.emit()

        # Start the saveData method in data_worker
        self.save_data_sig.emit()

    def closeEvent(self, event):
        self.stop_plot_sig.emit()
        event.accept()

    def switchSaveButtonStatus(self, status):
        """Sets state of save_button (Enabled/Disabled)
        Attributes:
        status (bool): To be state of button.
        """
        self.save_data_button.setEnabled(status)


class WarningWindow(QWidget):
    @staticmethod
    def showWarning(warning_title, warning_text, warning_info):
        warning_box = QMessageBox()
        warning_box.setIcon(QMessageBox.Icon.Warning)
        warning_box.setWindowTitle(warning_title)
        warning_box.setText(warning_text)
        warning_box.setInformativeText(warning_info)
        warning_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        warning_box.setDefaultButton(QMessageBox.StandardButton.Ok)
        warning_box.exec()
