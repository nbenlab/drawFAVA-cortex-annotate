# -*- coding: utf-8 -*-
################################################################################
# annotate/_viewer_panels.py

"""
Core implementation code for the CortexViewer's subpanels. 

CortexViewer is implemented in terms of two main subpanels:
1) CortexControlPanel: The control panel that contains viewer information about 
    the dataset, participant, and hemisphere, as well as user controls for 
    adjusting the inflation level of the cortex plot.
2) CortexFigurePanel: The figure panel that contains the 3D cortex plot.

Each panel is implemented as an ipywidget, and the CortexViewer class
assembles these two panels into a single interface.

CortexViewer is implmented in _viewer.py.
"""

# Imports ----------------------------------------------------------------------

import numpy as np
import k3d
import neuropythy as ny
import ipywidgets as ipw
from matplotlib.colors import to_rgb, to_rgba

# The Cortex Control Panel  ----------------------------------------------------

class CortexControlPanel(ipw.VBox):
    """Control Panel for the Cortex Viewer Tool.

    The panel that contains the information and user controls for the Cortex 
    Viewer tool.
    """

    def __init__(self, state):
        """Initialize the Cortex Control Panel with the given state."""

        # Create information boxes
        self.infobox = {} # initialize infobox dictionary
        for key in ( "dataset", "participant", "hemisphere", "annotation" ):
            self.infobox[key] = self._init_infobox(state, key)
        
        # Create the inflation slider widget
        self.inflation_slider = self._init_inflation_slider(state)

        # Create the overlay dropdown widget
        self.overlay_dropdown = self._init_overlay_dropdown(state)

        # Create the overlay alpha slider widget
        self.overlay_slider = self._init_overlay_slider(state)

        # Create point size slider (user-defined points)
        self.point_size_slider = self._init_point_size_slider(state)

        # Create line size slider (interpolated)
        self.line_size_slider = self._init_line_size_slider(state)

        # Create the number of line points slider widget
        self.line_points_slider = self._init_line_points_slider(state)

        # Assemble the control panel with children widgets
        children = [
            # HTML header
            self._make_html_header(),
            # Cortex Viewer title
            self._make_section_title("Selection:"),
            # Dataset label
            self.infobox["dataset"],
            # Participant label
            self.infobox["participant"],
            # Hemisphere label
            self.infobox["hemisphere"],
            # Annotation label
            self.infobox["annotation"],
            # Horizontal line
            self._make_hline(), 
            # Style Options title
            self._make_section_title("Style:"),
            # Inflation slider
            self.inflation_slider,
            # Overlay dropdown
            self.overlay_dropdown,
            # Overlay alpha slider
            self.overlay_slider, 
            # Point size slider
            self.point_size_slider, 
            # Line size slider
            self.line_size_slider,
            # Line interpolation slider
            self.line_points_slider,
        ]

        # Set the overall layout into the VBox
        super().__init__(children = children)

        # Add CSS class for styling
        self.add_class("cortex-control-panel")


    @classmethod
    def _make_html_header(cls):
        return ipw.HTML(f"""
            <style>
                .cortex-control-panel {{
                    background-color: #f0f0f0;
                    border: 1px solid lightgray;
                    padding: 2px;
                    height: 100%;
                    width: 260px; 
                }}
                .info-label {{
                    margin-right: 8px;
                    text-align: right;
                }}
                .info-value {{
                    background-color: white;
                    border: 1px solid rgb(158, 158, 158);
                    padding: 0px 8px 0px 8px;
                    text-align: left;
                }}
                .cortex-control-widgets {{
                    margin: 1% 3% 1% 3%;
                    width: 94%;
                }} 
                .cortex-control-widgets .widget-readout {{
                    min-width: 50px;
                }}
                .cortex-hline {{
                    border: 1px solid lightgray;
                    margin: 3%;
                    height: 0px;
                    width: 94%;
                }}
            </style>
        """)
    

    @classmethod
    def _make_section_title(cls, title):
        return ipw.HTML(f"<b style=\"margin: 0% 3% 0% 3%;\">{title}</b>")
    

    @classmethod
    def _make_hline(cls):
        return ipw.HTML("""<div class="cortex-hline"></div>""")


    def _prep_infobox_value(self, state, key):
        if key == "dataset":
            return state.dataset
        elif key == "participant":
            return state.participant
        elif key == "hemisphere":
            return state.convert_hemisphere(state.hemisphere)
        else: # key == "annotation"
            return state.selected_annotation


    def _make_infobox_value(self, value):
        """Make the infobox value for display."""
        return f"""<div class="info-value">{value}</div>"""
    

    def _init_infobox(self, state, key):
        """Update an information box for the given key and state."""
        value = self._prep_infobox_value(state, key)
        return ipw.Box([
            ipw.HTML(f"""<div class="info-label">{key.capitalize()}:</div>""",
                     layout = { "width": "40%", "margin": "0px" }),
            ipw.HTML(self._make_infobox_value(value), 
                     layout = { "width": "60%", "margin": "0px" }),
        ], layout = ipw.Layout(
            display = "flex",
            margin  = "1% 3% 1% 3%",
        ))


    def _init_inflation_slider(self, state):
        inflation_slider = ipw.IntSlider(
            value             = state.style["inflation_percent"],
            min               = 0,
            max               = 100,
            step              = 1,
            description       = "Inflation %:",
            continuous_update = False,
            orientation       = "horizontal",
            readout           = True,
            readout_format    = "d", 
        )
        inflation_slider.add_class("cortex-control-widgets")
        return inflation_slider


    def _init_overlay_dropdown(self, state):
        overlay_dropdown = ipw.Dropdown(
            options     = [
                ( "None", "curvature" ), 
                ( "Polar Angle", "angle" ), 
                ( "Eccentricity", "eccen" ), 
                ( "Variance Explained", "vexpl" )
            ],
            value       = state.style["overlay"],    
            description = "Overlay:",
        )
        overlay_dropdown.add_class("cortex-control-widgets")
        return overlay_dropdown


    def _init_overlay_slider(self, state):
        overlay_slider = ipw.FloatSlider(
            value             = state.style["overlay_alpha"],
            min               = 0.0,
            max               = 1.0,
            step              = 0.1,
            description       = "Alpha:",
            continuous_update = False,
            orientation       = "horizontal",
            readout           = True,
            readout_format    = ".1f", 
        )
        overlay_slider.add_class("cortex-control-widgets")
        return overlay_slider
    

    def _init_point_size_slider(self, state):
        point_size_slider = ipw.FloatSlider(
            value             = state.style["point_size"],
            min               = 0.5,
            max               = 5,
            step              = 0.1,
            description       = "Point Size:",
            continuous_update = False,
            orientation       = "horizontal",
            readout           = True,
            readout_format    = ".1f", 
        )
        point_size_slider.add_class("cortex-control-widgets")
        return point_size_slider
    

    def _init_line_size_slider(self, state):
        line_size_slider = ipw.FloatSlider(
            value             = state.style["line_size"],
            min               = 0.5,
            max               = 5,
            step              = 0.1,
            description       = "Line Size:",
            continuous_update = False,
            orientation       = "horizontal",
            readout           = True,
            readout_format    = ".1f", 
        )
        line_size_slider.add_class("cortex-control-widgets")
        return line_size_slider


    def _init_line_points_slider(self, state):
        line_points_slider = ipw.IntSlider(
            value             = state.style["line_points"],
            min               = 2,
            max               = 20,
            step              = 1,
            description       = "Line Points:",
            continuous_update = False,
            orientation       = "horizontal",
            readout           = True,
            readout_format    = "d", 
        )
        line_points_slider.add_class("cortex-control-widgets")
        return line_points_slider
    

    def refresh_infobox(self, state, key):
        """Refresh the control panel display to reflect updated infobox values."""
        value = self._prep_infobox_value(state, key)
        self.infobox[key].children[1].value = self._make_infobox_value(value)


    def observe_inflation_slider(self, callback): 
        """A method to register a callback for changes in the inflation slider."""
        self.inflation_slider.observe(callback, names = "value")


    def observe_overlay_dropdown(self, callback):
        """A method to register a callback for changes in the overlay dropdown."""
        self.overlay_dropdown.observe(callback, names = "value")

    
    def observe_overlay_slider(self, callback):
        """A method to register a callback for changes in the overlay alpha slider."""
        self.overlay_slider.observe(callback, names = "value")


    def observe_point_size_slider(self, callback):
        """A method to register a callback for changes in the point size slider."""
        self.point_size_slider.observe(callback, names = "value")


    def observe_line_size_slider(self, callback):
        """A method to register a callback for changes in the line size slider."""
        self.line_size_slider.observe(callback, names = "value")

    
    def observe_line_points_slider(self, callback):
        """A method to register a callback for changes in the line points slider."""
        self.line_points_slider.observe(callback, names = "value")

# The Cortex Figure Panel ------------------------------------------------------

class CortexFigurePanel(ipw.GridBox):
    """Cortex Figure Panel.

    The panel that contains the 3D cortex plot for the Cortex Viewer tool.
    """
    
    def __init__(
            self, 
            state, 
            width  = None,
            height = 400
        ):
        # Create a figure background
        self.figure = k3d.plot(
            #width   = width,  # K3d doesn't use width, just fills the space
            height=height,
            grid_visible=False,
            camera_auto_fit=True,
            camera_fov=1,
            axes_helper=0,
            menu_visibility=False,
        )
        
        # Draw the cortex plot (meshes) onto the k3d plot
        self.k3dmesh = k3d.mesh(
            state.mesh.coordinates.T.astype(np.float32),
            state.mesh.tess.indexed_faces.T.astype(np.uint32),
            flat_shading=False,
            wireframe=False,
            colors=self._rgb_to_k3dcolor(
                ny.graphics.cortex_plot_colors(state.mesh)))
        self.figure += self.k3dmesh
        #if state.hemisphere == 'lh':
        #    self.figure.camera = [
        #        4000.0, -9300.0, 800.0,
        #        0, 0, 0,
        #        0, 0, 1
        #    ]
        #else:
        #    self.figure.camera = [
        #        -4000.0, -9300.0, 800.0,
        #        0, 0, 0,
        #        0, 0, 1
        #    ]
        
        # Add scatter plots for annotations ( points and lines )
        self.k3dpoints_active = self._init_active_scatter(state)
        # Background annotation points
        self.k3dpoints_background = self._init_background_scatter(state)

        layout = ipw.Layout(
            grid_template_columns='1fr',
            height=f"{height}px",
            width=("100%" if width is None else "{width}px"))
        super().__init__([self.figure], layout=layout)

    def _rgb_to_k3dcolor(self, color, nrows=None):
        """Converts a matplorlib color into a color integer for k3d.

        If the given color is a matrix of RGB or RGBA triples, then a list of
        integers, one per row, is returned. If the given color is a list of
        strings or colors, then each is converted.
        """
        if nrows is not None:
            c = self._rgb_to_k3dcolor(color)
            if isinstance(c, int):
                return [c]*nrows
            elif len(c) == nrows:
                return c
            raise ValueError(f"{nrows} rows requested but {len(c)} produced")
        if isinstance(color, int):
            return color
        if isinstance(color, str):
            return self._rgb_to_k3dcolor(to_rgb(color))
        color = np.asarray(color)
        if np.issubdtype(color.dtype, np.floating):
            color = np.round(color * 255).astype(np.uint8)
        elif np.issubdtype(color.dtype, np.str_):
            return self._rgb_to_k3dcolor(list(map(to_rgb, color)))
        elif not np.issubdtype(color.dtype, np.uint8):
            raise ValueError("RGB arrays must be floating point or uint8")
        if len(color.shape) == 2:
            # Matrix of RGB or RGBA...
            if color.shape[1] == 4:
                if not np.all(color[:, -1] == 255):
                    raise ValueError("k3dcolors cannot include transparency")
                color = color[:, :3]
            elif color.shape[1] != 3:
                raise ValueError("color matrices must be N x 3 or N x 4")
            color = color.T
        elif len(color.shape) != 1:
            raise ValueError("can only convert lists and matrices of rgb")
        elif color.shape[0] != 3 and color.shape[0] != 4:
            raise ValueError("rgb or rgba values are required")
        (r,g,b) = color.astype(np.uint32)
        return ((r << 16) | (g << 8) | b)

    def _init_scatter(self):
        """Initialize an empty scatter plot."""
        #return ipv.scatter(0, 0, 0, visible = False, marker = "sphere")
        return k3d.points(
            positions=np.array([[0,0,0]], dtype=np.float32),
            point_size=0.5,
            shader='2d',
            color=0x8888ff
        )

    def _prep_active_scatter(self, state):
        """Prepare the data for the active annotation."""
        # Get the selected annotation
        selected_annotation = state.selected_annotation
        
        # Get the annotation style, coordinates, and anchor points
        annotation_style = state.annotation_styles[selected_annotation]
        coordinates = state.surface_annotations[selected_annotation]["coordinates"] 
        anchor      = state.surface_annotations[selected_annotation]["anchor"]

        # If not visible or no coordinates, return None
        if (not annotation_style["visible"]) or (coordinates is None):
            return None
                
        # Get number of annotation points
        n_points = coordinates.shape[1]

        # Prepare scatter sizes by anchor points
        scatter_sizes = np.ones(n_points) * state.style["line_size"]
        scatter_sizes[anchor == 1] = state.style["point_size"]

        # Prepare colors
        rgb_color = np.array(to_rgb(state.style_active["color"]))
        rgb_color = np.tile(rgb_color, (n_points, 1))

        # Return the scatter plot keyword arguments
        return { 
            "x"     : coordinates[0, :], 
            "y"     : coordinates[1, :], 
            "z"     : coordinates[2, :], 
            "size"  : scatter_sizes, 
            "color" : rgb_color 
        }


    def _prep_background_scatter(self, state):
        """Prepare the data for the background annotations."""
        # Get the list of annotations excluding the selected one
        selected_annotation = state.selected_annotation
        annotation_list = list(state.surface_annotations.keys())
        annotation_list.remove(selected_annotation)
        
        # Initialize arrays for all colors and coordinates
        all_colors = np.empty((0, 3)) 
        all_coordinates = np.empty((3, 0)) 

        for annotation in annotation_list: # for each annotation
            annotation_style = state.annotation_styles[annotation]
            coordinates = state.surface_annotations[annotation]["coordinates"] 

            if annotation_style["visible"] and coordinates is not None: 
                # Get colors for the annotation points
                rgb_color = np.array(to_rgb(annotation_style["color"]))
                rgb_color = np.tile(rgb_color, (coordinates.shape[-1], 1)) 

                # Concatenate coordinates and colors
                all_coordinates = np.hstack((all_coordinates, coordinates))
                all_colors = np.vstack((all_colors, rgb_color))

        # If no coordinates, return None
        if all_coordinates.shape[1] == 0:
            return None
        
        # Return the scatter plot keyword arguments
        return { 
            "x"     : all_coordinates[0,:], 
            "y"     : all_coordinates[1,:], 
            "z"     : all_coordinates[2,:], 
            "size"  : state.style["line_size"], 
            "color" : all_colors 
        }
            
    
    def _init_active_scatter(self, state):
        """Initialize the scatter plot for the active annotation."""
        scatter_kwargs = self._prep_active_scatter(state)
        if scatter_kwargs is None:
            points = self._init_scatter()
        else:
            points = k3d.points(
                np.transpose([scatter_kwargs[k] for k in ('x','y','z')]).astype(np.float32),
                shader='2d',
                point_size=float(scatter_kwargs['size']),
                color=int(self._rgb_to_k3dcolor(scatter_kwargs['color'][0]))
            )
        self.figure += points
        return points

    def _init_background_scatter(self, state):
        """Initialize the scatter plot for the background annotations."""
        scatter_kwargs = self._prep_background_scatter(state)
        if scatter_kwargs is None:
            points = self._init_scatter()
        else:
            points = k3d.points(
                np.transpose([scatter_kwargs[k] for k in ('x','y','z')]).astype(np.float32),
                shader='2d',
                point_size=float(scatter_kwargs['size']),
                color=int(self._rgb_to_k3dcolor(scatter_kwargs['color'][0]))
            )
        self.figure += points
        return points

    def refresh_figure(self, state):
        """Update the existing figure mesh coordinates and annotation points."""
        # Update the figure panel's mesh values
        self.k3dmesh.vertices = state.coordinates.T.astype(np.float32)
        self.k3dmesh.colors = self._rgb_to_k3dcolor(state.color)
        
        # Update the surface active annotation
        active_kwargs = self._prep_active_scatter(state)
        if active_kwargs is None:
            self.k3dpoints_active.visible = False
        else:
            coords = np.transpose([active_kwargs[k] for k in ('x','y','z')])
            self.k3dpoints_active.positions = coords.astype(np.float32)
            self.k3dpoints_active.size = active_kwargs["size"]
            self.k3dpoints_active.colors = self._rgb_to_k3dcolor(active_kwargs["color"], len(coords))
            self.k3dpoints_active.visible = True
        
        # Update the surface background annotation
        background_kwargs = self._prep_background_scatter(state)
        if background_kwargs is None:
            self.k3dpoints_background.visible = False
        else:
            coords = np.transpose([background_kwargs[k] for k in ('x','y','z')])
            self.k3dpoints_background.positions = coords.astype(np.float32)
            self.k3dpoints_background.size = background_kwargs["size"]
            self.k3dpoints_background.colors = self._rgb_to_k3dcolor(background_kwargs["color"], len(coords))
            self.k3dpoints_background.visible = True
