"""
This file is part of PressurePlotImproved (PIMP).
PressurePlotImproved is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
PressurePlotImproved is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Foobar. If not, see <https://www.gnu.org/licenses/>. 



Contact: Jacob Dannecker 1911196@stud.hs-mannheim.de (email valid until Feb 2026).
Additional for Lift calculation in worker.py by by: David Shell 
"""

""" workers.py """

from PyQt6.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import animation
import windows

import numpy as np
import pandas as pd

from itertools import product
from datetime import datetime
import logging
import time

from scipy.interpolate import Akima1DInterpolator


class Plot(QObject):
    """
    This class handles all plot related tasks like: setting up the plot, updating the
    plot, react on check boxes which alter the plot.

    Signals:
    start_data_sig (pyqtSignal): Start readDataLoop method in the data worker.
    deactivate_checkboxes_sig (pyqtSignal): Deactivate plot option check boxes
                                            in plot window.
    activate_checkboxes_sig (pyqtSignal):  Activate plot option check boxes
                                           in plot window.

    Slots:
    plot: Initializes and shows the plot.
    updateSetupData: Updates the instance attributes according to setup window.
    updateData: Updates instance attributes for pressure, temperature, density, etc.
    updateInfoLabel: Updates the info label in plot window.
    updateDemoLabel: Updates the Demo warning label in plot window.
    updateCheckboxTuple: Updates the plot options checkbox tuple.
    updateAngelVelocity: Updates the angel of attack and the velocity.
    updateLiftAndSplines: Upates the lift attribute and the splines

    Methods:
    rotateArrow: Rotates the arrow according to angle.
    startAnimation: Starts the animation.
    animate: Animation Function.
    plotPause: Does not alter the plot.
    plotPressures: Plots pressures.
    plotCp: Plots cp.
    setAxPressures: Sets plot properties to show pressures.
    setAxCp: Sets plot properties to show cp.
    updateDynamicLCDs: Updates the pyqt LCD widgets in plot window.

    """

    start_data_sig = pyqtSignal()
    deactivate_checkboxes_sig = pyqtSignal()
    activate_checkboxes_sig = pyqtSignal()
    switch_save_button_sig = pyqtSignal(bool)

    def __init__(self, DEFAULTS):
        """
        All Default variables are set, as well as the dictionaries created
        that contain the different mehtods which are used in the animate
        method.

        Attributes:
        self.default_label_color (str): Default color of info label
        self.plot_window (windows.PlotWindow): Plot window object form windows class.
        self.fig (matplotlib):  Plot figure.
        self.ax (matplotlib): Axis in plot figure.
        self.canvas (matplotlib): Canvas for plot.
        self.toolbar (matplotlib): Matplotlib toolbar.
        self.angle (float): Angle of attack.
        self.velocity (float): Free stream velocity.
        self.lift (float): Lift.
        self.pressure_array (np array): Stores all 16 pressure readings.
        self.mask_top (np array): Boolean mask to filter deactivated pressure ports.
        self.mask_bottom (np array): Boolean mask to filter deactivated pressure ports.
        self.checkbox_tuple (tuple): Booleans to indicate state of plot option
                                     check boxes.
        self.density_array (np array): Stores temperature, ambient pressure, humidity,
                                       density.
        self.temperature (float): Temperature.
        self.p_amb (float): Ambient pressure.
        self.humidity: Ambient humidity.
        self.density: Density.
        self.y_lim_max (int): Axis limits for pressure plot.
        self.y_lim_min (int): Axis limits for pressure plot.
        self.x_lim_max (int): Axis limits for pressure plot.
        self.x_lim_min (int): Axis limits for pressure plot.
        self.y_lim_min_cp (int): Axis limits for cp plot.
        self.y_lim_max_cp (int): Axis limits for cp plot.
        self.x_label (str): Label x-Axis.
        self.y_label_pressures (str): Label y-Axis pressure plot.
        self.y_label_cp (str): Label y-Axis cp plot.
        self.marker_pressures_top (str): Marker type for matplotlib.
        self.marker_pressures_bottom (str): Marker type for matplotlib.
        self.color_top (str): Color top markers.
        self.color_bottom (str): Color bottom markers.
        self.color_splines (str): Color splines.
        self.color_wing (str): Color wing.
        self.profile_name (str): Name of loaded profile.
        self.textbox_properties (dict): Properties of matplotlib textbox.
        self.textbox_coords (tuple): Coordinates of text box.
        self.y_wing_scale_pressures (int): Scale factor for wing in y direction
                                           pressure plot.
        self.y_wing_scale_cp (int): Scale factor for wing in x direction cp plot.
        self.linestyle_wing (str): Matplotlib line style of wing.
        self.alpha_wing (float): Transparency of wing.
        self.linewidth_wing (float): Line width of wing.
        self.arrow_props (dict): Properties of angle of attack arrow.
        self.arrow_length (float): Length of arrow.
        self.arrow_tip (np array): Position of arrow tip [x, y].
        self.arrow_tail (np array): Position of arrow tail [x,y].
        self.reverse_arrow_angle (bool): Reverse rotation of error.
        self.reverse_wing (bool): Reverse wing.
        self.arrow_sign (int): Either 1 or -1.
        self.wing_sign (int): Either 1 or - 1.
        self.lift_data_available (bool): Set by calculations worker, Default = False,
                                         makes sure that plot functions only plot
                                         the wing and splines if data is available
                                         to avoid crashing.
        self.plot_options_dict (dict): Containing the different methods for each plot
                                       option i.e pressure, pause or cp. The keys are
                                       tuples of boolean referring to the plot option
                                       check boxes.
        self.ax_options_dict (dict): Contains the different plot properties for pressure
                                     or cp plot.
        self.plot_init_flag (bool): Indicates that plot method already runing. Prevents
                                    multiple plots.

        Following vars only define the names. As soon as the calculation worker
        emits first results, the names refer to numpy arrays. This is to ensure
        the matplotlib objects don't crash, if the calculation worker is not fast enough
        at the startup of the application.

        self.x_wing_interpolation_top (empty list): List to store interpolation points
                                                    for wing.
        self.x_wing_interpolation_bottom (empty list): List to store interpolation points
                                                    for wing.
        self.y_wing_top (empty list): List to store interpolation points for wing.
        self.y_wing_bottom (empty list):  List to store interpolation points for wing.
        """

        super().__init__()
        self.plot_window = windows.PlotWindow(DEFAULTS)

        # Plot objects
        self.fig = Figure()
        self.ax = self.fig.add_subplot()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas)
        self.plot_window.v_layout.addWidget(self.toolbar)
        self.plot_window.v_layout.addWidget(self.canvas)

        # Initial Values for LCDs and spin boxes
        self.angle = DEFAULTS["ANGLE"]
        self.velocity = DEFAULTS["VELOCITY"]
        self.lift = DEFAULTS["LIFT"]
        self.pressure_array = DEFAULTS["PRESSURE_ARRAY"]
        self.mask_top = DEFAULTS["MASK_TOP"]
        self.mask_bottom = DEFAULTS["MASK_BOTTOM"]
        self.checkbox_tuple = DEFAULTS["CHECKBOX_TUPLE"]
        self.density_array = DEFAULTS["PRESSURE_ARRAY"]
        self.temperature = self.density_array[0]
        self.p_amb = self.density_array[1]
        self.humidity = self.density_array[2]
        self.density = self.density_array[3]

        # Plot defaults
        self.y_lim_max = DEFAULTS["Y_LIM_MAX_SCIEN"]
        self.y_lim_min = DEFAULTS["Y_LIM_MIN_SCIEN"]
        self.x_lim_max = DEFAULTS["X_LIM_MAX"]
        self.x_lim_min = DEFAULTS["X_LIM_MIN"]
        self.y_lim_min_cp = DEFAULTS["Y_LIM_MIN_CP"]
        self.y_lim_max_cp = DEFAULTS["Y_LIM_MAX_CP"]
        self.x_label = DEFAULTS["X_LABEL"]
        self.y_label_pressures = DEFAULTS["Y_LABEL_PRESSURES"]
        self.y_label_cp = DEFAULTS["Y_LABEL_CP"]
        self.marker_pressures_top = DEFAULTS["MARKER_PRESSURES_TOP"]
        self.marker_pressures_bottom = DEFAULTS["MARKER_PRESSURES_BOTTOM"]
        self.color_top = DEFAULTS["COLOR_TOP"]
        self.color_bottom = DEFAULTS["COLOR_BOTTOM"]
        self.color_splines = DEFAULTS["COLOR_SPLINES"]
        self.color_wing = DEFAULTS["COLOR_WING"]
        self.profile_name = DEFAULTS["PROFILE_NAME"]
        self.textbox_properties = DEFAULTS["TEXTBOX_PROPS"]
        self.textbox_coords = DEFAULTS["TEXTBOX_COORDS"]
        self.y_wing_scale_pressures = DEFAULTS["Y_WING_SCALE_PRESSURES"]
        self.y_wing_scale_cp = DEFAULTS["Y_WING_SCALE_CP"]
        self.linestyle_wing = DEFAULTS["LINESTYLE_WING"]
        self.alpha_wing = DEFAULTS["ALPHA_WING"]
        self.linewidth_wing = DEFAULTS["LINEWIDTH_WING"]
        self.arrow_props = DEFAULTS["AOA_ARROW_PROPS"]
        self.arrow_length = DEFAULTS["ARROW_LENGTH"]
        self.arrow_tip = DEFAULTS["ARROW_TIP"]
        self.arrow_tail = np.array([self.arrow_length, 0.0])

        self.reverse_arrow_angle = DEFAULTS["REVERSE_ARROW_ANGLE"]
        self.reverse_wing = DEFAULTS["REVERSE_WING"]

        if self.reverse_arrow_angle == True:
            self.arrow_sign = 1
        else:
            self.arrow_sign = -1

        if self.reverse_wing == True:
            self.wing_sign = -1
        else:
            self.wing_sign = 1

        self.x_wing_interpolation_top = []
        self.x_wing_interpolation_bottom = []
        self.y_wing_top = []
        self.y_wing_bottom = []
        self.lift_data_available = False

        # Options dictionaries
        # Generate all Tuples for pause (first entry is True)
        combinations = list(product([False, True], repeat=2))
        pause_tuples = [(True,) + combo for combo in combinations]
        self.plot_options_dict = {key: self.plotPause for key in pause_tuples}
        # Add other cases
        self.plot_options_dict[(False, True, True)] = self.plotCp
        self.plot_options_dict[(False, False, False)] = self.plotPressures
        self.plot_options_dict[(False, False, True)] = self.plotCp
        self.plot_options_dict[(False, True, False)] = self.plotPressures
        self.ax_options_dict = {
            (False,): self.setAxPressures,
            (True,): self.setAxCp,
        }

        self.updateDemoLabel("")
        self.plot_init_flag = False

        # Get default colot of info label
        color = self.plot_window.info_label.palette().color(
            self.plot_window.info_label.foregroundRole()
        )
        self.default_label_color = color.name()

    def rotateArrow(self):
        """
        Rotates the angle of attack arrow according to self.angle and aspect ratio.

        Attributes:
        self.arrow (matplotlib ax annotation): Shows angle of attack in plot.

        """
        x_lim = self.ax.get_xlim()
        y_lim = self.ax.get_ylim()
        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]
        aspect_ratio = y_range / x_range
        self.arrow_tail[0] = -1 * self.arrow_length * (np.cos(np.deg2rad(self.angle)))
        self.arrow_tail[1] = (
            self.arrow_sign
            * self.arrow_length
            * (np.sin(np.deg2rad(self.angle)))
            * 2
            * aspect_ratio
        )
        self.arrow.set_position(self.arrow_tail)

    def plot(self):
        """
        Initializes the plot, if self.plot_init_flag is set to False. Then emits the
        self.start_data_sig, shows plot window, starts the animation and sets
        self.plot_init_flag to True.
        If self.plot_init_flag is set to True when method is called only emits the
        self.start_data_sig and shows the plot window.
        This prevents the creation of multiple line2D objects, that otherwise would
        overlay in the plot.

        Attributes:
        self.y_lim_max (int): Axis limits for pressure plot.
        self.y_lim_min (int): Axis limits for pressure plot.
        self.x_lim_max (int): Axis limits for pressure plot.
        self.x_lim_min (int): Axis limits for pressure plot.
        self.y_lim_min_cp (int): Axis limits for cp plot.
        self.y_lim_max_cp (int): Axis limits for cp plot.
        self.x_label (str): Label x-Axis.
        self.y_label_pressures (str): Label y-Axis pressure plot.
        self.y_label_cp (str): Label y-Axis cp plot.
        self.marker_pressures_top (str): Marker type for matplotlib.
        self.marker_pressures_bottom (str): Marker type for matplotlib.
        self.color_top (str): Color top markers.
        self.color_bottom (str): Color bottom markers.
        self.color_splines (str): Color splines.
        self.color_wing (str): Color wing.
        self.profile_name (str): Name of loaded profile.
        self.textbox_properties (dict): Properties of matplotlib text box.
        self.textbox_coords (tuple): Coordinates of text box.
        self.y_wing_scale_pressures (int): Scale factor for wing in y direction
                                           pressure plot.
        self.y_wing_scale_cp (int): Scale factor for wing in x direction cp plot.
        self.linestyle_wing (str): Matplotlib line style of wing.
        self.alpha_wing (float): Transparency of wing.
        self.linewidth_wing (float): Linewidth of wing.
        self.line_pressure_top (line2D matplotlib): Top line for pressures.
        self.line_pressure_bottom (line2D matplotlib): Bottom line for pressures.
        self.line_spline_top (line2D matplotlib): Interpolated spline top.
        self.line_spline_bottom (line2D matplotlib): Interpolated spline bottom.
        self.line_wing_top (line2D matplotlib): Top line of wing.
        self.line_wing_bottom (line2D matplotlib): Bottom line of wing.
        self.plot_window (windows.PlotWindow): Plot window object form windows class.
        self.arrow (matplotlib ax annotation): Shows angle of attack in plot.
        self.artists (list): List of all line2D objects in self.ax
        self.plot_init_flag (bool): Indicates that plot method already running. Prevents
                                    multiple plots.
        Signals:
        start_data_sig (pyqtSignal): Start readDataLoop method in the data worker.
        """
        if not self.plot_init_flag:
            self.start_data_sig.emit()
            self.plot_window.show()

            self.ax.set_xlim(self.x_lim_min, self.x_lim_max)
            self.ax.set_ylim(self.y_lim_min, self.y_lim_max)
            self.ax.axhline(y=0, color="k", linewidth=0.7)
            self.ax.set_xlabel(self.x_label)
            self.ax.set_ylabel(self.y_label_pressures)
            (self.line_pressures_top,) = self.ax.plot(
                [],
                [],
                color=self.color_top,
                linestyle="none",
                marker=self.marker_pressures_top,
            )

            (self.line_pressures_bottom,) = self.ax.plot(
                [],
                [],
                color=self.color_bottom,
                linestyle="none",
                marker=self.marker_pressures_bottom,
            )
            (self.line_spline_top,) = self.ax.plot(
                [], [], color=self.color_splines, marker="none"
            )
            (self.line_spline_bottom,) = self.ax.plot(
                [], [], color=self.color_splines, marker="none"
            )
            (self.line_wing_top,) = self.ax.plot(
                [],
                [],
                color=self.color_wing,
                marker="none",
                alpha=self.alpha_wing,
                linestyle=self.linestyle_wing,
                linewidth=self.linewidth_wing,
            )
            (self.line_wing_bottom,) = self.ax.plot(
                [],
                [],
                color=self.color_wing,
                marker="none",
                alpha=self.alpha_wing,
                linestyle=self.linestyle_wing,
                linewidth=self.linewidth_wing,
            )

            self.ax.text(
                self.textbox_coords[0],
                self.textbox_coords[1],
                self.profile_name,
                verticalalignment="top",
                transform=self.ax.transAxes,
                bbox=self.textbox_properties,
                fontsize=10,
            )

            # AoA Arrow
            self.arrow = self.ax.annotate(
                "",
                xy=self.arrow_tip,
                xycoords="data",
                xytext=self.arrow_tail,
                textcoords="data",
                arrowprops=self.arrow_props,
            )

            # Make list of all Line2D objects
            self.artists = self.ax.get_lines()
            self.artists.append(self.arrow)
            self.canvas.draw()

            # Prevent multiple animations
            self.startAnimation()
            self.setAxPressures()
            self.plot_init_flag = True

        else:
            self.start_data_sig.emit()
            self.updateInfoLabel(("Pressure Plot Improved :-)", "default"))
            self.switch_save_button_sig.emit(True)
            self.plot_window.show()

    def startAnimation(self):
        """Create a matplotlib FuncAnimation object"""
        self.ani = animation.FuncAnimation(
            self.fig, self.animate, interval=10, blit=True, cache_frame_data=False
        )

    def animate(self, frame):
        """
        Animation Function, gets called by the self.ani animation object. Depending on
        the self.checkbox_tuple, the line2D objects will be updated by different
        methods e.g. plotPressure or plotCp.

        Attributes:
        self.plot_options_dict (dict): Containing the different methods for each plot
                                       option i.e pressure, pause or cp. The keys are
                                       tuples of boolean referring to the plot option
                                       check boxes.
        Returns:
        self.artists (list): List of all line2D objects in self.ax
        """
        self.plot_options_dict.get(self.checkbox_tuple)()
        return self.artists

    def plotPause(self):
        """
        Part of plot_options_dict. Does not update the line2D objects contained in
        self.artists. Deactivates all plot options check boxes, except the pause
        checkbox in the plot window.
        Signals:
        deactivate_checkboxes_sig (pyqtSignal): Deactivate plot option checkb oxes in
                                                plot window.
        """
        self.deactivate_checkboxes_sig.emit()

    def plotPressures(self):
        """
        Part of plot_options_dict. Updates the line2D objects with pressure values.

        Attributes:
        self.pressure_array (np array): Stores all 16 pressure readings.
        self.x_coordinates_top (np array): x-coords of pressure points top.
        self.x_coordinates_bottom np_array): x-coords of pressure points bottom.
        self.mask_top (np array): Boolean mask to filter deactivated pressure ports.
        self.mask_bottom (np array): Boolean mask to filter deactivated pressure ports.
        self.checkbox_tuple (tuple): Booleans to indicate state of plot option
                                     checkboxes.
        self.y_wing_scale_pressures (int): Scale factor for wing in y direction
                                           pressure plot.
        self.wing_sign (int): Either 1 or - 1.
        self.lift_data_available (bool): Set by calculations worker, Default = False,
                                         makes sure that plot functions only plot
                                         the wing and splines if data is available
                                         to avoid crashing.
        self.x_wing_interpolation_top (np array): List to store interpolation points
                                                    for wing.
        self.x_wing_interpolation_bottom (np array): List to store interpolation points
                                                    for wing.
        self.y_wing_top (np array): List to store interpolation points for wing.
        self.y_wing_bottom (np array):  List to store interplation points for wing.
        self.line_pressure_top (line2D matplotlib): Top line for pressures.
        self.line_pressure_bottom (line2D matplotlib): Bottom line for pressures.
        self.line_spline_top (line2D matplotlib): Interpolated spline top.
        self.line_spline_bottom (line2D matplotlib): Interpolated spline bottom.
        self.line_wing_top (line2D matplotlib): Top line of wing.
        self.line_wing_bottom (line2D matplotlib): Bottom line of wing.
        self.x_grid_interpolation (np array): Interpolated points for splines.
        self.spline_top (np array): y-values of spline top.
        self.spline_bottom (np array): y-values of spline bottom.

        Signals:
        self.activate_checkboxes_sig: Activates the plot option check boxes.

        Methods:
        self.updateDynamicLCDs: Updates the LCDs in plot window.
        """

        self.activate_checkboxes_sig.emit()
        # Update pressures
        self.line_pressures_top.set_data(
            self.x_coordinates_top,
            self.pressure_array[self.mask_top],
        )
        self.line_pressures_bottom.set_data(
            self.x_coordinates_bottom,
            self.pressure_array[self.mask_bottom],
        )
        # Update Splines

        if self.checkbox_tuple[1] == True and self.lift_data_available:
            self.line_spline_top.set_data(self.x_grid_interpolation, self.spline_top)
            self.line_spline_bottom.set_data(
                self.x_grid_interpolation, self.spline_bottom
            )
            # Wing
            self.line_wing_top.set_data(
                self.x_wing_interpolation_top / self.x_wing_interpolation_top[-1],
                self.wing_sign * self.y_wing_top * self.y_wing_scale_pressures,
            )
            self.line_wing_bottom.set_data(
                self.x_wing_interpolation_bottom / self.x_wing_interpolation_bottom[-1],
                self.wing_sign * self.y_wing_bottom * self.y_wing_scale_pressures,
            )


        else:
            # Erase Splines
            self.line_spline_top.set_data([], [])
            self.line_spline_bottom.set_data([], [])

        self.updateDynamicLCDs()

    def plotCp(self):
        """
        Part of plot_options_dict. Updates the line2D objects with cp values.

        Attributes:
        self.pressure_array (np array): Stores all 16 pressure readings.
        self.x_coordinates_top (np array): x-coords of pressure points top.
        self.x_coordinates_bottom np_array): x-coords of pressure points bottom.
        self.mask_top (np array): Boolean mask to filter deactivated pressure ports.
        self.mask_bottom (np array): Boolean mask to filter deactivated pressure ports.
        self.checkbox_tuple (tuple): Boolean to indicate state of plot option
                                     check boxes.
        self.y_wing_scale_cp (int): Scale factor for wing in y direction
                                           cp plot.
        self.wing_sign (int): Either 1 or - 1.
        self.lift_data_available (bool): Set by calculations worker, Default = False,
                                         makes sure that plot functions only plot
                                         the wing and splines if data is available
                                         to avoid crashing.
        self.x_wing_interpolation_top (np array): List to store interpolation points
                                                    for wing.
        self.x_wing_interpolation_bottom (np array): List to store interpolation points
                                                    for wing.
        self.y_wing_top (np array): List to store interpolation points for wing.
        self.y_wing_bottom (np array):  List to store interpolation points for wing.
        self.line_pressure_top (line2D matplotlib): Top line for pressures.
        self.line_pressure_bottom (line2D matplotlib): Bottom line for pressures.
        self.line_spline_top (line2D matplotlib): Interpolated spline top.
        self.line_spline_bottom (line2D matplotlib): Interpolated spline bottom.
        self.line_wing_top (line2D matplotlib): Top line of wing.
        self.line_wing_bottom (line2D matplotlib): Bottom line of wing.
        self.x_grid_interpolation (np array): Interpolated points for splines.
        self.spline_top (np array): y-values of spline top.
        self.spline_bottom (np array): y-values of spline bottom.
        denominator_cp (float): Denominator to calculate cp.

        Signals:
        self.activate_checkboxes_sig: Activates the plot option check boxes.

        Methods:
        self.updateDynamicLCDs: Updates the LCDs in plot window.
        """

        self.activate_checkboxes_sig.emit()
        # Update pressures
        denominator_cp = 0.5 * self.velocity**2 * self.density
        self.line_pressures_top.set_data(
            self.x_coordinates_top,
            self.pressure_array[self.mask_top] / denominator_cp,
        )
        self.line_pressures_bottom.set_data(
            self.x_coordinates_bottom,
            self.pressure_array[self.mask_bottom] / denominator_cp,
        )

        # Update Splines
        if self.checkbox_tuple[1] == True:
            self.line_spline_top.set_data(
                self.x_grid_interpolation,
                self.spline_top / denominator_cp,
            )

            self.line_spline_bottom.set_data(
                self.x_grid_interpolation,
                self.spline_bottom / denominator_cp,
            )
            # Wing
            self.line_wing_top.set_data(
                self.x_wing_interpolation_top / self.x_wing_interpolation_top[-1],
                self.wing_sign * self.y_wing_top * self.y_wing_scale_cp,
            )
            self.line_wing_bottom.set_data(
                self.x_wing_interpolation_bottom / self.x_wing_interpolation_bottom[-1],
                self.wing_sign * self.y_wing_bottom * self.y_wing_scale_cp,
            )


        else:
            # Erase Splines
            self.line_spline_top.set_data([], [])
            self.line_spline_bottom.set_data([], [])

            self.updateDynamicLCDs()

    def setAxPressures(self):
        """
        Updates the ax to display pressure values.
        self.fig (matplotlib):  Plot figure.
        self.ax (matplotlib): Axis in plot figure.

        Methods:
        self.rotateArrow: Rotates the arrow according to angle.
        """

        self.ax.set_ylim(self.y_lim_min, self.y_lim_max)
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label_pressures)
        self.fig.canvas.draw_idle()
        self.rotateArrow()

    def setAxCp(self):
        """
        Updates the ax to display cp values.

        Attributes:
        self.fig (matplotlib):  Plot figure.
        self.ax (matplotlib): Axis in plot figure.

        Methods:
        self.rotateArrow: Rotates the arrow according to angle.
        """
        self.ax.set_ylim(self.y_lim_min_cp, self.y_lim_max_cp)
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label_cp)
        self.fig.canvas.draw_idle()
        self.rotateArrow()

    @pyqtSlot(tuple)
    def updateCheckboxTuple(self, checkbox_tuple):
        """Slot that updates the checkbox_tuple variable and calls"""
        self.checkbox_tuple = checkbox_tuple
        # Update the ax properties according to check boxes
        self.ax_options_dict.get((self.checkbox_tuple[2],))()
        self.fig.canvas.draw()

    @pyqtSlot(tuple)
    def updateInfoLabel(self, information_tuple):
        """Update the info label in plot window with new information and color"""
        information = information_tuple[0]
        color = information_tuple[1]
        if color == "default":
            color_style = "color: " + self.default_label_color + ";"
            self.plot_window.info_label.setStyleSheet(color_style)
        else:
            color_style = "color: " + color + ";"
            self.plot_window.info_label.setStyleSheet(color_style)
        self.plot_window.info_label.setText(information)

    @pyqtSlot(str)
    def updateDemoLabel(self, text):
        """Updates the demo label with information string"""
        self.plot_window.demo_label.setStyleSheet("color: red;")
        self.plot_window.demo_label.setText(text)

    def updateDynamicLCDs(self):
        """Updates all dynamically updated LCDs in plot window"""
        # Update all LCD Labels to newest Data
        self.plot_window.temp_lcd.display(f"{self.temperature:.02f}")
        self.plot_window.p_amb_lcd.display(f"{self.p_amb:.02f}")
        self.plot_window.hum_lcd.display(f"{self.humidity:.01f}")
        self.plot_window.dens_lcd.display(f"{self.density:.02f}")
        self.plot_window.lift_top_lcd.display(f"{self.lift:.02f}")
        self.plot_window.lift_bottom_lcd.display(f"{self.lift:.02f}")

    @pyqtSlot(tuple)
    def updateData(self, data):
        """Updates the instance variables with new data"""
        self.pressure_array = data[0]
        density_array = data[1]
        self.temperature = density_array[0]
        self.p_amb = density_array[1]
        self.humidity = density_array[2]
        self.density = density_array[3]

    @pyqtSlot(tuple)
    def updateLiftAndSplines(self, lift_and_splines):
        """Updates the instance variables with new data form calculations worker"""
        self.lift = lift_and_splines[0]
        self.spline_top = lift_and_splines[1]
        self.spline_bottom = lift_and_splines[2]
        self.x_grid_interpolation = lift_and_splines[3]
        self.y_wing_top = lift_and_splines[4]
        self.y_wing_bottom = lift_and_splines[5]
        self.x_wing_interpolation_top = lift_and_splines[6]
        self.x_wing_interpolation_bottom = lift_and_splines[7]
        self.lift_data_available = lift_and_splines[8]

    @pyqtSlot(tuple)
    def updateAngleVelocity(self, spinbox_values):
        """
        Slot that updates the instance variables referring to the spin boxes in the plot
        window. Is called, when the values in said spin boxes change.
        """
        self.angle = spinbox_values[0]
        self.velocity = spinbox_values[1]
        self.rotateArrow()

    @pyqtSlot(tuple)
    def updateSetupData(self, setup_data):
        """
        Slot gets called, when setup window is closed with "OK"". Updates all instance
        variable containing data from setup window.
        """
        self.mask_top = setup_data[0]
        self.mask_bottom = setup_data[1]
        self.x_coordinates = setup_data[2]
        self.x_coordinates_top = self.x_coordinates[self.mask_top]
        self.x_coordinates_bottom = self.x_coordinates[self.mask_bottom]


class Data(QObject):
    """
    This class handles all task related to get new data and save data to a file.

    Signals:
    send_data_sig (pyqtSignal): Emits new data to all connected slots.
    calc_lift_and_splines (pyqtSignal): Triggers the calculation of lift and splines.
    update_info_label_sig (pyqtSignal): Triggers update of the info label in plot window.
    update_demo_label_sig (pyqtSignal): Triggers update of demo label.
    switch_plot_button_sig (pyqtSignal): Set new state of plot button in main window.
    switch_save_button_sig (pyqtSignal): Set new state of save_data_button in plot window.

    Slots:
    updateSerialConnection: Updates a serial connection to device.
    readDataLoop: Starts a loop that continuously reads data.
    updateSetupData: Updates the setup data from setup window.
    stopDataLoop: Stops the readData loop.
    updatePath: Updates the file where data is stored.
    updateSpinboxValues: Updates the instance variables containing the spin box values of
                         plot window.
    updateLift: Updates the instance variable containing the calculated lift.
    saveData: Takes a mean of x measurements and saves them to a file.

    Methods:
    readData: Reads data via serial connection.
    sendData: Calls send_data_sig.
    """

    send_data_sig = pyqtSignal(tuple)
    calc_lift_splines_sig = pyqtSignal()
    update_info_label_sig = pyqtSignal(tuple)
    update_demo_label_sig = pyqtSignal(str)
    switch_plot_button_sig = pyqtSignal(bool)
    switch_save_button_sig = pyqtSignal(bool)

    def __init__(self, DEFAULTS):
        """
        All Default variables are set.

        Attributes
        self.stop_flag (bool): Signal that read data shall stop when True.
        self.path (str): Stores path to file where data shall be written.
        self.number_of_measurements (int): Number of measurements for mean.
        self.angle (float): Anlge of attack.
        self.velocity (float): Free stream velocity.
        self.width (float): Widht of wing.
        self.header (list): Contains headers for csv file.
        self.lift (float): Calculated lift of wing.
        self.demo_mode (bool): Indicates whether demo mode is active.
        self.demo_pressure (np array): Pressures for demo mode.
        self.demo_density (np array): Density for demo mode.
        self.demo_pressure_factor (float): Scales random pressure fluctuation.
        self.demo_density_factor (float): Scales random density fluctuation.
        """

        super().__init__()
        self.stop_flag = False
        self.path = None
        self.number_of_measurements = DEFAULTS["NUM_OF_MEASSUREMENTS"]
        self.angle = DEFAULTS["ANGLE"]
        self.velocity = DEFAULTS["VELOCITY"]
        self.width = DEFAULTS["WIDTH"]
        self.header = DEFAULTS["HEADER"]
        self.lift = DEFAULTS["LIFT"]
        self.demo_mode = DEFAULTS["DEMO_MODE"]
        self.demo_pressure = DEFAULTS["DEMO_PRESSURE"]
        self.demo_density = DEFAULTS["DEMO_DENSITY"]
        self.demo_pressure_factor = DEFAULTS["DEMO_PRESSURE_FACTOR"]
        self.demo_density_factor = DEFAULTS["DEMO_DENSITY_FACTOR"]

    @pyqtSlot(tuple)
    def updateSerialConnections(self, serial_connections):
        """Updates serial connections"""
        self.pressure_device, self.density_device = serial_connections

    @pyqtSlot()
    def readDataLoop(self):
        """Starts loop, that continuously reads data from serial connections"""
        try:
            # Sends the Data to the plot worker
            self.stop_flag = False
            while self.stop_flag == False:
                self.readData()
                self.sendData()
                self.calc_lift_splines_sig.emit()
        except BaseException as err:
            print("ERROR in method readDataLoop.")
            self.update_info_label_sig.emit(
                (
                    "There was an Error getting new Data from devices! Close window and rerun setup!",
                    "red",
                )
            )
            self.switch_plot_button_sig.emit(False)
            self.switch_save_button_sig.emit(False)
            logging.error(str(err), exc_info=True)

    def readData(self):
        try:
            if self.demo_mode:
                self.update_demo_label_sig.emit("DEMO MODE!")
                self.pressure_array = (
                    self.demo_pressure
                    + np.random.random(16) * self.demo_pressure_factor
                )
                self.density_array = (
                    self.demo_density + np.random.rand(4) * self.demo_density_factor
                )
                time.sleep(0.1)
            else:
                self.pressure_array = self.pressure_device.getNewData()
                self.density_array = self.density_device.getNewData()

        except BaseException as err:
            print(err)
            logging.error(str(err), exc_info=True)
            raise Exception("ERROR in method readData.")

    @pyqtSlot(tuple)
    def updateSetupData(self, setup_data):
        """Updates instance variables storing data from setup window"""
        self.mask_top = setup_data[0]
        self.mask_bottom = setup_data[1]
        self.x_coordinates = setup_data[2]
        self.x_coordinates_top = self.x_coordinates[self.mask_top]
        self.x_coordinates_bottom = self.x_coordinates[self.mask_bottom]

    def sendData(self):
        """Emits Data to all connected slots"""
        self.send_data_sig.emit((self.pressure_array, self.density_array))

    def stopDataLoop(self):
        """Stops readData loop by setting stop_flag True"""
        self.stop_flag = True

    @pyqtSlot(tuple)
    def updatePath(self, path):
        """Updates path where data shall be saved"""
        self.path = path[0]

    @pyqtSlot(tuple)
    def updateSpinboxValues(self, spinbox_values):
        """Updates instance variables storing values form spin boxes in plot window"""
        self.angle = spinbox_values[0]
        self.velocity = spinbox_values[1]
        self.width = spinbox_values[2]

    def updateLift(self, lift_and_splines):
        """Updates instance variable storing calculated lift"""
        self.lift = lift_and_splines[0]

    @pyqtSlot()
    def saveData(self):
        """
        Takes a certain amount of measurements, calculates the mean value and stores
        them in a csv file.

        Attributes:
        start_time (float): Time when measurements start.
        now_time (float): Now time.
        total_time (float): Total time of measurements.
        num_pressures (np array): Measurements to calculate mean of.
        num_density (np array): Measurements to calculate mean of.
        num_lift (np array): Calculation Results to calculate mean of.
        pressures_mean (np array): Mean pressures.
        lift_mean (float): Mean lift.
        df_time (pd dataframe): Stores date and time.
        df_additional_data (pd dataframe): Additional data to be saved.
        df_pressures (pd dataframe): Mean pressures.
        df_density (pd dataframe): Mean density data.
        df_x_coords (pd dataframe): x-coodinates of measurement points.
        mask_top_str (str): Status top points x-coords.
        mask_bottom_str (str): Status bottom x-coords.
        df_x_masks (pd dataframe): Concatenation of mask_top, mask_bottom.
        df_new (pd dataframe): Concatenation of all dataframes.
        df_old (pd dataframe): Last dataframe from file, empty if file is empty
        df_write (pd dataframe): Dataframe that is written to file.

        Signals:
        self.update_info_label_sig: Updates the info label in plot window.
        Mehtods:
        self.readDataLoop: Starts the readData loop.
        """

        try:
            start_time = time.time()
            # Create array to store data in
            num_pressures = np.zeros((self.number_of_measurements, 16))
            num_density = np.zeros((self.number_of_measurements, 4))
            num_lift = np.zeros(self.number_of_measurements)

            for i in range(self.number_of_measurements):
                self.calc_lift_splines_sig.emit()
                self.readData()
                self.sendData()
                num_pressures[i, :] = self.pressure_array
                num_density[i, :] = self.density_array
                num_lift[i] = self.lift

            pressures_mean = np.mean(num_pressures, axis=0)
            lift_mean = np.mean(num_lift)

            # density-array: [Temp., Amb., Hum., Dens.]
            density_mean = np.mean(num_density, axis=0)
            now_time = time.time()
            total_time_measurements = now_time - start_time
            # Assemble Data to single dataframe
            now_time = datetime.now().strftime("%H:%M:%S")
            df_time = pd.DataFrame([now_time])
            df_additional_data = pd.DataFrame(
                [
                    [
                        self.angle,
                        self.velocity,
                        lift_mean,
                        self.width,
                        self.number_of_measurements,
                        total_time_measurements,
                    ]
                ]
            )
            df_pressures = pd.DataFrame([pressures_mean])
            df_density = pd.DataFrame([density_mean])
            df_x_coords = pd.DataFrame([self.x_coordinates])
            # Convert mask_array to binary string ex. [True, False] -> "10"
            mask_top_str = "".join(map(str, self.mask_top.astype(np.int_)))
            mask_bottom_str = "".join(map(str, self.mask_bottom.astype(np.int_)))

            df_x_masks = pd.DataFrame([[mask_top_str, mask_bottom_str]])
            df_new = pd.concat(
                (
                    df_time,
                    df_density,
                    df_additional_data,
                    df_pressures,
                    df_x_coords,
                    df_x_masks,
                ),
                axis=1,
            )

            # Check whether the file exists and append df_new accordingly
            if self.path.is_file():
                df_old = pd.read_csv(self.path)
                df_new.columns = self.header
                df_write = pd.concat([df_old, df_new], ignore_index=True)
                df_write.to_csv(self.path, index=False)
            else:
                df_new.columns = self.header
                df_new.to_csv(self.path, index=False)
            self.update_info_label_sig.emit(
                (f"{now_time} : Data saved in {self.path}", "default")
            )
        except BaseException as err:
            print(err)
            self.update_info_label_sig.emit(("ERROR in method saveData", "red"))
            logging.error(str(err), exc_info=True)

        self.readDataLoop()


class Calculations(QObject):
    """
    This class handles all long running calculations.

    Signals:
    send_lift_splines_sig (pyqtSignal): Sends calculation results and information.

    Slots:
    updateSetupData: Updates the setup data from setup window.
    updateData: Updates instance attributes for pressure, temperature, density, etc.
    updateSpinboxValues: Updates the instance variables containing the spin box values of
                         plot window.
    calculateLiftandSplines: Calculates the lift and the splines.

    """

    send_lift_splines_sig = pyqtSignal(tuple)

    def __init__(self, DEFAULTS):
        super().__init__()
        # Default data
        self.pressure_array = DEFAULTS["PRESSURE_ARRAY"]
        self.angle = DEFAULTS["ANGLE"]
        self.width = DEFAULTS["WIDTH"]
        self.chord_length = DEFAULTS["CHORD_LENGTH"]
        self.x_wing_top = DEFAULTS["X_WING_TOP"]
        self.x_wing_bottom = DEFAULTS["X_WING_BOTTOM"]
        self.y_wing_top = DEFAULTS["Y_WING_TOP"]
        self.y_wing_bottom = DEFAULTS["Y_WING_BOTTOM"]
        self.number_of_interpolated_points = DEFAULTS["NUM_INTERPOL_POINTS"]
        self.spline_top = np.zeros(self.number_of_interpolated_points)
        self.spline_bottom = np.zeros(self.number_of_interpolated_points)

    @pyqtSlot(tuple)
    def updateSetupData(self, setup_data):
        """
        Slot gets called, when setup window is closed with "OK"". Updates all instance
        variable containing data from setup window.
        """

        self.mask_top = setup_data[0]
        self.mask_bottom = setup_data[1]
        self.x_coordinates = setup_data[2]
        self.x_coordinates_top = self.x_coordinates[self.mask_top]
        self.x_coordinates_bottom = self.x_coordinates[self.mask_bottom]

    @pyqtSlot(tuple)
    def updateData(self, data):
        """Updates the instace variables with new data"""

        self.pressure_array = data[0]
        density_array = data[1]
        self.temperature = density_array[0]
        self.p_amb = density_array[1]
        self.humidity = density_array[2]
        self.density = density_array[3]

    @pyqtSlot(tuple)
    def updateSpinboxValues(self, spinbox_values):
        """Updates instance variables storing values form spin boxes in plot window"""

        self.angle = spinbox_values[0]
        self.velocity = spinbox_values[1]
        self.width = spinbox_values[2]

    @pyqtSlot()
    def calculateLiftAndSplines(self):
        """
        Calculates lift and splines using the instance variables. At the end emits new
        calculation results to all connected slots. This code is mainly written by
        David Schell and was only changed slightly to fit the needs of this applications.
        Attributes:
        pressures_top (np array): Pressure on top of wing.
        pressures_bottom (np array): Pressures on bottom of wing.
        x_coordinates_top_calc (np array): x-coordinates top.
        x_coordinates_bottom_calc (np array): x-coordinates bottom.
        last_pressure_bottom (float): Last pressure of pressures_top.
        last_pressure_top (float): Last pressure of pressures_bottom.
        x_grid_interpolation (np array):
        interpolate_pressures_top (np array):
        interpolate_pressures_bottom (np array):
        self.spline_top (instance variable)
        self.spline_bottom
        self.number_of_interpolated_points
        x_wing_interpolation_top
        x_wing_interpolation_bottom
        self.x_wing_top
        self.x_wing_bottom
        delta_x_top
        delta_x_bottom
        delta_y_top
        delta_y_bottom
        alpha_top (np array): Angles between interpolated points.
        alpha_bottom (np array) Angles between interpolated points.
        area_top (np array): Areas between interpolated points.
        area_bottom (np array) Areas between interpolated points.
        norm_y_top (np array): Normal vector on areas.
        norm_y_bottom (np array): Normal vector on areas.
        lift_top (float): Lift top surface.
        lift_bottom (float): Lift bottom surface.
        lift  (flaot): Sum of lift top and bottom.
        # Acount for angle of attack and round for two decimals
        self.lift (float): Instance variable for lift.
        lift_data_available (bool): Set by calculations worker, Default = False,
                                    makes sure that plot functions only plot
                                    the wing and splines if data is available
                                    to avoid crashing.

        Signals:
        self.send_lift_splines_sig (pyqtSignal): Sends calculation results and
                                                 information.
        """
        try:
            # Get latest pressures top
            pressures_top = self.pressure_array[self.mask_top]
            pressures_bottom = self.pressure_array[self.mask_bottom]
            x_coordinates_top_calc = self.x_coordinates_top
            x_coordinates_bottom_calc = self.x_coordinates_bottom

            # Sort in ascending order, so that interpolation can work

            sorted_indices_top = np.lexsort((pressures_top, x_coordinates_top_calc))
            x_coordinates_top_calc = x_coordinates_top_calc[sorted_indices_top]
            pressures_top = pressures_top[sorted_indices_top]

            sorted_indices_bottom = np.lexsort(
                (pressures_bottom, x_coordinates_bottom_calc)
            )
            x_coordinates_bottom_calc = x_coordinates_bottom_calc[sorted_indices_bottom]
            pressures_bottom = pressures_bottom[sorted_indices_bottom]

            # Following if-statement check whether the first and last
            # pressure are close to 0 and 1. If not extra points (0, 0)
            # and (1, linear interpolate) get appended/prepended.
            # This ensures an interpolation all over the wing.

            if x_coordinates_top_calc[-1] < 0.99:
                last_pressure_top = pressures_top[-2] + (
                    (pressures_top[-1] - pressures_top[-2])
                )
                pressures_top = np.append(pressures_top, last_pressure_top)
                x_coordinates_top_calc = np.append(x_coordinates_top_calc, 1)

            if x_coordinates_bottom_calc[-1] < 0.99:
                last_pressure_bottom = pressures_bottom[-2] + (
                    (pressures_bottom[-1] - pressures_bottom[-2])
                    / (self.x_coordinates_bottom[-1] - self.x_coordinates_bottom[-2])
                ) * (1 - self.x_coordinates_bottom[-2])
                pressures_bottom = np.append(pressures_bottom, last_pressure_bottom)
                x_coordinates_bottom_calc = np.append(x_coordinates_bottom_calc, 1)

            if x_coordinates_top_calc[0] > 0.00001:
                pressures_top = np.append(0, pressures_top)
                x_coordinates_top_calc = np.append(0, x_coordinates_top_calc)

            if x_coordinates_bottom_calc[0] > 0.00001:
                pressures_bottom = np.append(0, pressures_bottom)
                x_coordinates_bottom_calc = np.append(0, x_coordinates_bottom_calc)

            # Set x-grid for interpolation of spline points
            x_grid_interpolation = np.linspace(0, 1, self.number_of_interpolated_points)

            # Interpolate pressures
            interpolate_pressures_top = Akima1DInterpolator(
                x_coordinates_top_calc, pressures_top
            )(x_grid_interpolation)
            interpolate_pressures_bottom = Akima1DInterpolator(
                x_coordinates_bottom_calc, pressures_bottom
            )(x_grid_interpolation)
            # Update splines
            self.spline_top = interpolate_pressures_top
            self.spline_bottom = interpolate_pressures_bottom

            # Interpolate points on wing geometry
            x_wing_interpolation_top = np.linspace(
                self.x_wing_top[0],
                self.x_wing_top[-1],
                self.number_of_interpolated_points,
            )
            x_wing_interpolation_bottom = np.linspace(
                self.x_wing_bottom[0],
                self.x_wing_bottom[-1],
                self.number_of_interpolated_points,
            )

            y_wing_top = Akima1DInterpolator(self.x_wing_top, self.y_wing_top)(
                x_wing_interpolation_top
            )

            y_wing_bottom = Akima1DInterpolator(self.x_wing_bottom, self.y_wing_bottom)(
                x_wing_interpolation_bottom
            )

            # Calculate deltas between interpolated points of wing geometry
            delta_x_top = np.diff(x_wing_interpolation_top * self.chord_length)
            delta_x_bottom = np.diff(x_wing_interpolation_bottom * self.chord_length)
            delta_y_top = np.diff(y_wing_top)
            delta_y_bottom = np.diff(y_wing_bottom)

            # Calculate angles between interpolated points
            alpha_top = np.arctan(delta_y_top / delta_x_top)
            alpha_bottom = np.arctan(delta_y_bottom / delta_x_bottom)

            # Calculate areas between interpolated points
            area_top = self.width * (delta_x_top / np.cos(alpha_top))
            area_bottom = self.width * (delta_x_bottom / np.cos(alpha_bottom))

            # Calculate normal vector on areas
            norm_y_top = -1 * np.cos(alpha_top)
            norm_y_bottom = np.cos(alpha_bottom)

            # Calculate lift
            lift_top = -1 * interpolate_pressures_top[1:] * norm_y_top * area_top
            lift_top = np.sum(lift_top)

            lift_bottom = (
                -1 * interpolate_pressures_bottom[1:] * norm_y_bottom
            ) * area_bottom
            lift_bottom = np.sum(lift_bottom)

            lift = lift_top + lift_bottom

            # Account for angle of attack and round for two decimals
            self.lift = lift * np.cos(np.deg2rad(self.angle))
            lift_data_available = True
            # Send calculated lift and splines
            self.send_lift_splines_sig.emit(
                (
                    self.lift,
                    self.spline_top,
                    self.spline_bottom,
                    x_grid_interpolation,
                    y_wing_top,
                    y_wing_bottom,
                    x_wing_interpolation_top,
                    x_wing_interpolation_bottom,
                    lift_data_available,
                )
            )
        except BaseException as err:
            print(err)
            logging.error(str(err), exc_info=True)
