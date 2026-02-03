# -*- coding: utf-8 -*-
################################################################################
# annotate/_viewer.py

"""
Implementation code for the Cortex Viewer.
"""

# Imports ----------------------------------------------------------------------

import glob
import numpy as np
import os.path as op
import neuropythy as ny
import ipywidgets as ipw
from struct import unpack
from functools import partial
from neuropythy.geometry.util import barycentric_to_cartesian

from ._viewer_panels import CortexControlPanel, CortexFigurePanel

# The Cortex Viewer State ------------------------------------------------------

class CortexViewerState:
    """
    The state object for the Cortex Viewer tool.
    """
    
    __FLATMAP_KWARGS = {
        "mask"      : ( "parcellation", 43 ), 
        "map_right" : "right", 
        "radius"    : np.pi / 2
    }

    __PROPERTY_KWARGS = {
        "angle" : { "func": lambda prop, h: { f"polar_angle_{h}": prop }, 
                     "kwargs": { } },
        "eccen" : { "func": lambda prop, _: { "eccentricity": prop }, 
                     "kwargs": { } },
        "vexpl" : { "func": lambda prop, _: prop, 
                     "kwargs": { "cmap": "hot", "vmin": 0, "vmax": 100 } },
    }


    def __init__(
            self, 
            annotation_widgets, 
            dataset_directory = "/home/jovyan/datasets", 
        ):
        
        # Store intialization variables
        self.annotation_widgets = annotation_widgets 
        self.dataset_directory  = dataset_directory

        # Prepare fsaverage data 
        self.fsaverage = self._get_fsaverage()

        # Prepare initial values from annotation widget
        self.dataset_index = self.get_selected_dataset_index()
        self.dataset       = self.get_selected_dataset()
        self.participant   = self.get_selected_participant()
        self.hemisphere    = self.get_selected_hemisphere() 
        self.selected_annotation = self.get_selected_annotation() 

        # Prepare initial annotation style from annotation widget
        self.update_style_options() 

        # Set cortex viewer style (default) options
        self.style = {
            "inflation_percent" : 100,
            "overlay"           : "curvature",
            "overlay_alpha"     : 1.0, 
            "point_size"        : 1.0, 
            "line_size"         : 0.5,
            "line_points"       : 10,
        }

        # Prepare participant 3d coordinate and mesh data
        self.update_participant()
        self.update_coordinates()
        self.update_mesh()
        self.update_color()

        # Extract the flatmap and annotation data
        self.update_flatmap_annotations()
        self.update_annotation_styles() 

        # Prepare surface annotations and path 
        self.update_surface_annotations() 


    def _get_fsaverage(self):
        """load fsaverage tesselations and inflated surface coordinates."""
        # Load the fsaverage object
        fsa = ny.freesurfer_subject("fsaverage")
        
        # Return fsaverage dictionary values
        return {
            h: {
                "tesselation": fsa.hemis[h].tess.faces, 
                "inflated"   : fsa.hemis[h].surface("inflated").coordinates, 
                "flatmap"    : fsa.hemis[h].mask_flatmap(
                    **CortexViewerState.__FLATMAP_KWARGS)
            } for h in ( "lh", "rh" )
        }


    def _get_active_annotation_tool(self):
        """Get the active multicanvas widget."""
        return self.annotation_widgets.children[self.dataset_index]


    def _get_active_selection_panel(self):
        """Get the active selection panel widget."""
        active_widget = self._get_active_annotation_tool()
        return active_widget.control_panel.selection_panel
    

    def _get_active_figure_panel(self):
        """Get the active figure panel widget."""
        active_widget = self._get_active_annotation_tool()
        return active_widget.figure_panel
    

    def get_selected_dataset_index(self):
        """Get the current dataset selection index."""
        return self.annotation_widgets.selected_index
    

    def get_selected_dataset(self):
        """Get the current dataset selection widget."""
        return self.annotation_widgets.titles[self.dataset_index]
    
    
    def get_selected_participant(self):
        """Get the current participant selection widget."""
        active_selection = self._get_active_selection_panel()
        return active_selection.children[0].value
    
    
    @staticmethod
    def convert_hemisphere(hemisphere):
        """Convert hemisphere string to code or vice versa."""
        str_to_code = { "Left Hemisphere"  : "lh", "Right Hemisphere" : "rh" }
        code_to_str = { v: k for k, v in str_to_code.items() }
        if hemisphere in str_to_code.keys():
            return str_to_code[hemisphere]
        elif hemisphere in code_to_str.keys():
            return code_to_str[hemisphere]
        else: 
            raise ValueError(f"Invalid hemisphere value: {hemisphere}")
    

    def get_selected_hemisphere(self):
        """Get the current hemisphere selection widget."""
        active_selection = self._get_active_selection_panel()
        hemisphere = active_selection.children[1].value
        return self.convert_hemisphere(hemisphere)
    

    def get_selected_annotation(self):
        """Get the current annotation selection widget."""
        active_selection = self._get_active_selection_panel()
        return active_selection.children[2].value


    def get_style_annotation(self):
        """Get the current style annotation selection widget."""
        return self.style_panel.style_dropdown.value
    
    
    def update_style_options(self):
        """Update the style panel options based on current state."""
        active_widget     = self._get_active_annotation_tool()
        self.style_panel  = active_widget.control_panel.style_panel
        self.styler       = active_widget.state.style
        self.style_active = active_widget.state.config.display.fg_options


    def update_flatmap_annotations(self):
        """Get the annotation tool's annotation dictionary"""
        figure_panel = self._get_active_figure_panel()
        self.flatmap_annotations = figure_panel.annotations    


    def update_annotation_styles(self, annotation = None):
        """Get the annotation tool's flatmap style dictionary"""
        # Initialize annotation styles dictionary if not present
        if hasattr(self, "annotation_styles") is False:
            self.annotation_styles = {}
        
        # Define annotations to update
        annotation_list = [annotation] if annotation is not None \
            else list(self.flatmap_annotations.keys())
        
        # Convert each flatmap annotation to surface paths
        for key in annotation_list: # for each annotation
            self.annotation_styles[key] = self.styler(key)


    def _read_coordinates(self, filename):
        """Read cortical mesh coordinates from a <hemisphere>.3d.coordinates file."""
        with open(filename, "rb") as f:
            values = f.read() # load file content
            fstr = "e" * (len(values) // 2)
            coordinates = np.array(unpack(fstr, values)).reshape((3, -1))
        return coordinates
    

    def _read_property(self, filename):
        """Read cortical property data from a <hemisphere>.3d.<property> file."""
        with open(filename, "rb") as f:
            values = f.read() # load file content
            fstr = "e" * (len(values) // 2)
            prop = np.array(unpack(fstr, values)).reshape((-1, ))
        return prop
    

    def update_participant(self): 
        """Load participant dataset based on current state."""
        # Locate participant directory and files
        participant_dir = op.join(
            self.dataset_directory, self.dataset.lower(), self.participant)
        filenames = glob.glob(op.join(participant_dir, f"{self.hemisphere}.3d.*"))

        # Load participant midgray coordinates
        coordinates_file = [x for x in filenames if x.endswith("coordinates")][0]
        self.midgray = self._read_coordinates(coordinates_file)

        # Load remaining property files
        property_files = [x for x in filenames if x != coordinates_file]
        self.properties = {
            fname.split(".")[-1]: self._read_property(fname)
            for fname in property_files
        }
    

    def update_coordinates(self):
        """Update the cortical mesh coordinates based on the inflation value."""
        inflated_coordinates = self.fsaverage[self.hemisphere]["inflated"]
        self.coordinates = ((inflated_coordinates - self.midgray) * \
             (self.style["inflation_percent"] / 100.0)) + self.midgray


    def update_mesh(self):
        """Update the cortical mesh object based on the current state."""
        self.mesh = ny.geometry.Mesh(
            faces       = self.fsaverage[self.hemisphere]["tesselation"],
            coordinates = self.coordinates,
            properties  = self.properties
        )
    

    def update_color(self):
        """Update the cortical mesh color based on curvature."""
        if self.style["overlay"] == "curvature":
            mesh_color = ny.graphics.cortex_plot_colors(self.mesh)[:, :3]
        else: # Get property kwargs and values 
            overlay_style = self.style["overlay"]
            prop_kwargs   = CortexViewerState.__PROPERTY_KWARGS[overlay_style]
            prop_values   = self.mesh.properties[overlay_style]
            prop_color    = prop_kwargs["func"](prop_values, self.hemisphere)
            mesh_color = ny.graphics.cortex_plot_colors(self.mesh, 
                color = prop_color, alpha = self.style["overlay_alpha"], 
                **prop_kwargs["kwargs"])[:, :3]
        self.color = mesh_color


    @staticmethod
    def _flatmap_to_surface(flatmap_address, mesh_coordinates):
        """Convert flatmap annotation coordinates to surface coordinates."""
        bary_faces  = flatmap_address["faces"]       # (3, n_faces)
        bary_coords = flatmap_address["coordinates"] # (2, n_points)
        tx = np.transpose(mesh_coordinates[:, bary_faces], (1, 0, 2)) # (3, 3, n_points)
        return barycentric_to_cartesian(tx, bary_coords) # (3, n_points)

    
    def _interpolate_coordinates(self, coordinates):
        """Interpolate coordinates along the path."""
        # Get number of interpolated points
        n = self.style["line_points"] + 2

        # Intialize ararys to store interpolated coordinates
        x_interp = []; y_interp = []

        # Iterate over each segment and interpolate points  
        n_interp = coordinates.shape[0] - 1 
        for i in np.arange(n_interp): # for each pair of coordinates
            xs, xe = coordinates[i, 0], coordinates[i+1, 0]
            ys, ye = coordinates[i, 1], coordinates[i+1, 1]
            x_interp.append(np.linspace(xs, xe, n))
            y_interp.append(np.linspace(ys, ye, n))

        # Concatenate and prepare interpolated points
        x_interp = np.concatenate(x_interp)
        y_interp = np.concatenate(y_interp)
        anchor   = np.tile(np.hstack([1, np.zeros((n - 2,)), 1]), n_interp)

        # Return unique interpolated coordinates
        interp_coordinates = np.vstack((x_interp, y_interp, anchor)).T
        interp_coordinates = np.unique(interp_coordinates, axis = 0)
        return interp_coordinates[:,:-1], interp_coordinates[:,-1]


    def update_surface_annotations(self, annotation = None): 
        """Update cortical surface annotation coordinates."""
        # Initialize surface annotations dictionary if not present
        if hasattr(self, "surface_annotations") is False:
            self.surface_annotations = {}

        # Get current fsaverage hemisphere flatmap
        fsa_flatmap = self.fsaverage[self.hemisphere]["flatmap"]

        # Define annotations to update
        annotation_list = [annotation] if annotation is not None \
            else list(self.flatmap_annotations.keys())
        
        # Convert each flatmap annotation to surface coordinates
        for key in annotation_list: # for each annotation
            # get the current roi's flatmap coorindates
            flatmap_coordinates = self.flatmap_annotations[key]

            # If no flatmap coordinates, set surface annotation to None
            if (flatmap_coordinates is None) or \
                (flatmap_coordinates.shape[0] == 0): 
                self.surface_annotations[key] = {
                    "coordinates" : None,
                    "anchor"      : None,
                }
            else: 
                # If multiple flatmap coordinates, interpolate first
                anchor = np.array([1]) # initailize
                if flatmap_coordinates.shape[0] > 1:
                    flatmap_coordinates, anchor = \
                        self._interpolate_coordinates(flatmap_coordinates)
                
                # Convert flatmap coordinates to addresses
                flatmap_address = fsa_flatmap.address(flatmap_coordinates)

                # Calculate surface coordinates
                surface_coordinates = self._flatmap_to_surface(
                    flatmap_address, self.coordinates)
            
                # Store surface annotation coordinates
                self.surface_annotations[key] = {
                    "coordinates" : surface_coordinates,
                    "anchor"      : anchor,
                }


    def observe_dataset_index(self, callback):
        """Assign a callback function to dataset value changes."""
        self.annotation_widgets.observe(callback, names = "selected_index")


    def observe_participant(self, callback):
        """Assign a callback function to participant value changes."""
        for annotation_widget in self.annotation_widgets.children:
            participant_dropdown = annotation_widget.control_panel.selection_panel.children[0]
            participant_dropdown.observe(callback, names = "value")


    def observe_hemisphere(self, callback):
        """Assign a callback function to hemisphere value changes."""
        for annotation_widget in self.annotation_widgets.children:
            hemisphere_dropdown = annotation_widget.control_panel.selection_panel.children[1]
            hemisphere_dropdown.observe(callback, names = "value")

    
    def observe_annotation(self, callback):
        """Assign a callback function to annotation value changes."""
        for annotation_widget in self.annotation_widgets.children:
            annotation_dropdown = annotation_widget.control_panel.selection_panel.children[2]
            annotation_dropdown.observe(callback, names = "value")
    

    def observe_annotation_styles(self, callback):
        """Assign a callback function to style option changes."""
        for annotation_widget in self.annotation_widgets.children:
            style_panel = annotation_widget.control_panel.style_panel
            style_panel.visible_checkbox.observe(callback, names = "value")
            style_panel.color_picker.observe(callback, names = "value")


    def observe_annotation_change(self, callback):
        """Assign a callback function to annotation data changes."""
        for annotation_widget in self.annotation_widgets.children:
            annotation_figure = annotation_widget.figure_panel
            annotation_figure.observe(callback, names = "_annotation_change")


# The Cortex Viewer Widget -----------------------------------------------------

class CortexViewer(ipw.GridBox):
    """The Cortex Viewer widget.

    The `CortexViewer` type handles the 3D Cortex figure that renders the 
    cortical mesh that assists the flatmap viewer.
    """

    def __init__(self, annotation_widgets, dataset_directory, panel_width=270):
        # Initialize the Cortex Viewer state
        self.state = CortexViewerState(
            annotation_widgets = annotation_widgets,
            dataset_directory  = dataset_directory,
        )

        # Create the Cortex Viewer control panel
        self.control_panel = CortexControlPanel(self.state)

        # Create the Cortex Viewer figure panel
        self.figure_panel = CortexFigurePanel(self.state)

        # Initialize the GridBox with the control panel and figure panel
        super().__init__(
            children = [self.control_panel, self.figure_panel], 
            layout   = ipw.Layout(
                border="1px solid rgb(158, 158, 158)", 
                padding="15px",
                grid_template_columns=f'{panel_width}px 1fr',
                grid_template_rows='auto'
            )
        )

        # Assign information box observers
        for k in self.control_panel.infobox.keys():
            self._infobox_observers[k](partial(self.on_selection_change, k))

        # Assign user annotation input observers
        self.state.observe_annotation_change(self.on_annotation_change)

        # Assign flatmap annotation style observers
        self.state.observe_annotation_styles(self.on_annotation_style_change)

        # Assign style option observers
        for k in self.state.style.keys():
            self._style_observers[k](partial(self.on_style_change, k)) 


    @property
    def _infobox_observers(self):
        """Return a list of observer functions for the Cortex Viewer state."""
        return {
            "dataset"     : self.state.observe_dataset_index,
            "participant" : self.state.observe_participant,
            "hemisphere"  : self.state.observe_hemisphere,
            "annotation"  : self.state.observe_annotation,
        }   


    @property
    def _style_observers(self):
        """Return a list of observer functions for the Cortex Viewer style."""
        return {
            "inflation_percent" : self.control_panel.observe_inflation_slider, 
            "overlay"           : self.control_panel.observe_overlay_dropdown, 
            "overlay_alpha"     : self.control_panel.observe_overlay_slider, 
            "point_size"        : self.control_panel.observe_point_size_slider, 
            "line_size"         : self.control_panel.observe_line_size_slider,
            "line_points"       : self.control_panel.observe_line_points_slider, 
        }
    

    def on_selection_change(self, key, change):
        """Handle changes to the dataset selection."""
        # Update the control panel information
        if key == "dataset":
            self.state.dataset_index = change.new
            self.state.dataset       = self.state.get_selected_dataset()
            self.state.participant   = self.state.get_selected_participant()
            self.state.hemisphere    = self.state.get_selected_hemisphere() 
            self.state.selected_annotation = self.state.get_selected_annotation()
        elif key == "participant":
            self.state.participant = change.new 
        elif key == "hemisphere":
            self.state.hemisphere = self.state.get_selected_hemisphere()
        elif key == "annotation": 
            self.state.selected_annotation = change.new
        
        # Update style options based on new dataset
        if key == "dataset":
            self.state.update_style_options()

        # Update the cortex viewer state based on selection change
        if key in ( "dataset", "participant", "hemisphere" ):
            self.state.update_participant()
            self.state.update_coordinates()
            self.state.update_mesh()
            self.state.update_color()
            self.state.update_flatmap_annotations()
            self.state.update_annotation_styles()
            self.state.update_surface_annotations()
        
        # Update the infobox displays
        for k in self.control_panel.infobox.keys():
            self.control_panel.refresh_infobox(self.state, k)

        # Refresh the figure with updated state
        self.figure_panel.refresh_figure(self.state)


    def on_annotation_style_change(self, change):
        """Handle changes to the flatmap style option changes."""
        # Get the annotation style annotation name
        annotation_name = self.state.get_style_annotation()
        if annotation_name != "Selected Annotation":
            # Update the annotation styles for the current annotation
            self.state.update_annotation_styles(annotation_name) 
            
            # Refresh the figure with updated state
            self.figure_panel.refresh_figure(self.state)
        
            
    def on_annotation_change(self, _):
        """Handle changes to the annotation data."""
        # Update surface annotations and paths for the current annotation
        self.state.update_surface_annotations(self.state.selected_annotation)

        # Refresh the figure with updated state
        self.figure_panel.refresh_figure(self.state)


    def on_style_change(self, key, change):
        """Handle changes to the style option changes."""
        # Change style based on key value
        self.state.style[key] = change.new

        # Update the mesh color based on the new overlay
        if key == "inflation_percent":
            self.state.update_coordinates()
            self.state.update_mesh()
            self.state.update_color()
            self.state.update_surface_annotations()
        elif key in ( "overlay", "overlay_alpha" ):
            self.state.update_color()
        elif key == "line_points":
            self.state.update_surface_annotations()
 
        # Update the figure with updated state
        self.figure_panel.refresh_figure(self.state)
