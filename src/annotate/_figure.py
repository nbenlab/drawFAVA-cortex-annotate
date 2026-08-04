# -*- coding: utf-8 -*-
################################################################################
# annotate/_figure.py

"""
Core implementation code for the cortex-annotate tool's figure panel.
"""


# Imports ######################################################################

import threading
import numpy as np
import ipycanvas as ipc
import ipywidgets as ipw
import matplotlib as mpl
from traitlets import Int
from collections import defaultdict

from ._util import wrap as wordwrap

# The Figure Panel #############################################################

class FigurePanel(ipw.HBox):
    """The canvas that manages the display of figures and annotations.

    The `FigurePanel` is an subclass of `ipycanvas.MultiCanvas` that is designed
    to manage the display of images and annotations for the `AnnotationTool` in
    `_core.py`.
    """

    class LoadingContext:
        """A context manager for the loading screen on the figure panel canvas."""
        __slots__ = ( "canvas", "message" )

        _count = defaultdict(lambda: 0)

        def __init__(self, canvas, message = "Loading..."):
            self.canvas  = canvas
            self.message = message


        def __enter__(self):
            count = FigurePanel.LoadingContext._count
            idc   = id(self.canvas)
            c = count[idc]
            if c == 0:
                FigurePanel._draw_loading(self.canvas, self.message)
            count[idc] = c + 1 


        def __exit__(self, type, value, traceback):
            count = FigurePanel.LoadingContext._count
            idc = id(self.canvas)
            c = count[idc]
            c -= 1
            count[idc] = c
            if c == 0:
                self.canvas.clear()
                del count[idc]


    # A traitlet that increments whenever the annotations change.
    _annotation_change = Int(default_value = 0)


    def __init__(self, state):
        # Store the state.
        self.state = state
        
        # Store the annotation configuation information.
        self.annot_cfg = state.config.annotations

        # Store the figure size (in pixels, cell in grid). 
        sz = state.preferences["figure_size"] 
        self.figure_size = np.array([sz, sz]) 

        # Get first grid shape from first annotation in state
        annot0 = list(self.annot_cfg.types.keys())[0]
        self.grid_shape = self.annot_cfg.grid_shape[annot0]

        # Calculate the canvas size (in pixels) from the figure size and grid shape.
        self.canvas_size = self.figure_size * self.grid_shape

        # Make a multicanvas.
        canvas_width, canvas_height = self.canvas_size
        self.multicanvas = ipc.MultiCanvas(
            5, width = canvas_width, height = canvas_height)

        # We always seem to need to explicitly set the layout size in pixels.
        self.multicanvas.layout.width  = f"{canvas_width}px"
        self.multicanvas.layout.height = f"{canvas_height}px"

        # Separate out the canvas layers.
        self.image_canvas      = self.multicanvas[0] # grid image layer
        self.background_canvas = self.multicanvas[1] # background annotation layer 
        self.active_canvas     = self.multicanvas[2] # active annotation layer
        self.loading_canvas    = self.multicanvas[3] # loading screen layer
        self.message_canvas    = self.multicanvas[4] # message layer (for errors, etc.)

        # Draw the loading screen on the loading canvas and save it as the loading context.
        self._draw_loading(self.loading_canvas)
        self.loading_canvas.save()
        self.loading_context = FigurePanel.LoadingContext(self.loading_canvas)

        # Set up our event observers for mouse clicks (to add points).
        self.multicanvas.on_mouse_down(self.on_mouse_click)

        # Set up our event observers for key presses (tab, delete).
        self.multicanvas.on_key_down(self.on_key_press)

        # Initialize the image variables.
        self.image = None
        self.grid  = None
        self.xlim  = None
        self.ylim  = None

        # Initialize the annotation variables.
        self.target      = None
        self.active      = None
        self.annotations = {}
        self.fixed_heads = {}
        self.fixed_tails = {}
        self.editable    = np.array([])
        self.cursor      = None

        # Initialize our parent class.
        super().__init__([ self._make_html_header(), self.multicanvas ])


    @classmethod
    def _make_html_header(cls):
        return ipw.HTML(f"""
            <style> 
                canvas {{
                    cursor: crosshair !important;
                }} 
            </style>
        """)

    # Image Canvas Methods -----------------------------------------------------
    
    def redraw_image(self):
        """Clears the image canvas and redraws the image."""
        with ipc.hold_canvas():
            # Clear the image canvas.
            self.image_canvas.clear()
            
            # Redraw the image
            self.image_canvas.draw_image(
                self.image, 0, 0, 
                self.image_canvas.width, 
                self.image_canvas.height
            )

    # Annotation Canvas Methods ------------------------------------------------

    def redraw_annotations(self, active = True, background = True):
        """Clears the annotation canvas and redraws all annotations."""
        # First, we clear the annotation canvases, depending on updates.
        if active: self.active_canvas.clear()
        if background: self.background_canvas.clear()

        # We step through all (visible) annotations and draw them.
        for (annotation, points) in self.annotations.items():
            # If there are no points, we can skip.
            if points is None or len(points) == 0: continue
            
            # Determine annotation specific properties based on if the current
            # annotation is the active annotation or a background annotation.
            if self.active == annotation:
                # Skip active annotation if active is False.
                if not active: continue
                canvas     = self.active_canvas
                styletag   = None
                cursor     = self.cursor
            else:
                # Skip background annotations if background is False.
                if not background: continue
                canvas     = self.background_canvas
                styletag   = annotation
                cursor     = None

            # Determine if the head or tail of this annotation is fixed.
            fixed_head = self.fixed_heads.get(annotation, None) is not None
            fixed_tail = self.fixed_tails.get(annotation, None) is not None

            # Get the style for this annotation.
            style = self.state.style(styletag) 

            # If this annotation isn't visible, we can skip it also.
            if not style["visible"]: continue
            
            # Check annotation type to see if the path is closed. Only the 
            # boundary type is closed.
            atype  = self.annot_cfg.types[annotation]
            closed = atype == "boundary" 
            
            # Okay, points needs to be drawn, so convert the figure points
            # into canvas coordinates (repeated across panels).
            grid_points = self.figure_to_canvas(points)

            # For all the point-matrices here, we need to draw them.
            for points in grid_points:
                self.draw_points(
                    canvas     = canvas, 
                    points     = points, 
                    style      = style,
                    cursor     = cursor, 
                    closed     = closed, 
                    fixed_head = fixed_head, 
                    fixed_tail = fixed_tail
                )
    
    
    def _apply_linestyle(self, canvas, style):
        """Applies the given line width and line style to the given canvas."""
        # Get the line width and line style from the style dict, with defaults.
        linewidth, linestyle = style["linewidth"], style["linestyle"]
        
        # Apply the line width and line style to the canvas.
        canvas.line_width = linewidth if linewidth is not None else 1
        if linestyle == "solid":
            canvas.set_line_dash([])
        elif linestyle == "dashed":
            canvas.set_line_dash([linewidth * 3, linewidth * 3])
        elif linestyle == "dot-dashed":
            canvas.set_line_dash([linewidth * 1, linewidth * 2, 
                                  linewidth * 4, linewidth * 2])
        elif linestyle == "dotted":
            canvas.set_line_dash([linewidth, linewidth])
        else:
            raise ValueError(f"Invalid linestyle: {linestyle}")
        

    def draw_points(
        self, canvas, points, style, cursor = None, closed = False, 
        fixed_head = False, fixed_tail = False
    ):
        """Draws the given path on the given canvas using the named style.

        `state.draw_path(name, path, canvas)` applies the style for the named
        annotation then draws the given `path` on the given `canvas`. Note that
        the `path` coordinate must be in canvas pixel coordinates, not figure
        coordinates.

        If the optional argument `path` is `False`, then only the points are
        drawn.

        If the optional argument `style` is given, then the given style dict
        is used instead of the stling for the `ann_name` annotation.
        """
        # Convert the color from the style into an RGB.
        rgb_color = np.array(mpl.colors.to_rgb(style["color"]))
        rgb_color = np.array(rgb_color * 255, dtype = np.uint8)

        # Apply the line width and line style to the canvas.
        self._apply_linestyle(canvas, style)

        # We only draw line segments if there are at least two points, and if
        # there are fixed points, we only draw when there are more points than
        # fixed points.
        if points.shape[0] > np.max([1, np.sum([fixed_head, fixed_tail])]):
            # if the path is closed, we need to add the first point to the end 
            # of the point matrix to make sure the path is closed when we draw it.
            if closed: points = np.vstack([points, points[0:1, :]])

            # create segement coordinates pairs [(x1, y1), (x2, y2), ...] 
            segments = np.stack([points[:-1,:], points[1:,:]], axis = 1)

            # draw the line segments for this path
            canvas.stroke_styled_line_segments(
                points = segments,
                color  = [rgb_color],  
            )

        # If fixed head, separate fixed points from the user points drawn.
        user_points  = points.copy()
        fixed_points = self.empty_point_matrix()
        if fixed_head: 
            fixed_points = np.vstack([fixed_points, points[0, :]])
            user_points  = user_points[1:, :] # remove the fixed head point 

        # If fixed tail, separate fixed poins from the points to be drawn.
        if fixed_tail:
            fixed_points = np.vstack([fixed_points, points[-1, :]])
            user_points  = user_points[:-1, :] # remove the fixed tail point
      
        # If there is at least one fixed point, we draw them in a separate call.
        if fixed_points.shape[0] > 0:
            canvas.fill_styled_rects(
                x      = fixed_points[:,0] - style["markersize"], 
                y      = fixed_points[:,1] - style["markersize"], 
                width  = style["markersize"] * 2,
                height = style["markersize"] * 2,
                color  = [rgb_color],
            )

        # If there is at least one point, we can draw a circle for each point.
        if user_points.shape[0] > 0:
            canvas.fill_styled_circles(
                x      = user_points[:,0], 
                y      = user_points[:,1], 
                radius = style["markersize"],
                color  = [rgb_color],
            )
        
        # If there is a cursor, we draw a larger circle around the point at the
        # cursor position to indicate that it is active.
        if cursor is not None: 
            active_point = points[cursor, :]
            canvas.stroke_styled_circles(
                x      = [active_point[0], ], 
                y      = [active_point[1], ], 
                radius = (style["markersize"] + 1) * 4/3, 
                color  = [rgb_color],
            )

    # Loading Canvas Methods ---------------------------------------------------

    @staticmethod
    def _prep_canvas_message(canvas, message, wrap = True, fontsize = 32):
        """Prepares a message for drawing on the given canvas."""
        # Prepare the message by word wrapping, if necessary.
        if wrap is True or wrap is Ellipsis:
            wrap = int(canvas.width * 13/15 / fontsize * 2)
        message = wordwrap(message, wrap = wrap)
    
        # Calculate the x0, y0, and max_width for the message.
        x0 = canvas.width // 15
        y0 = canvas.height // 15
        max_width = canvas.width - (canvas.width // 15 * 2)
        
        # Return the prepared message and the x0, y0, and max_width for drawing it.
        return message, x0, y0, max_width
    

    @staticmethod
    def _draw_text_canvas(canvas, message, wrap = True, fontsize = 32):
        """Draws a message on the given canvas."""
        # Prepare the message by word wrapping, if necessary.
        message, x0, y0, max_width = FigurePanel._prep_canvas_message(
            canvas, message, wrap = wrap, fontsize = fontsize)
        
        with ipc.hold_canvas():
            # Clear the canvas.
            canvas.clear()

            # Draw a white background with some transparency.
            canvas.fill_style   = "white"
            canvas.global_alpha = 0.85
            canvas.fill_rect(0, 0, canvas.width, canvas.height)
            
            # Draw the message in black.
            canvas.fill_style    = "black"
            canvas.global_alpha  = 1
            canvas.font          = f"{fontsize}px HelveticaNeue"
            canvas.text_align    = "left"
            canvas.text_baseline = "top"

            # Draw the message on the canvas.
            for (i, line) in enumerate(message.split("\n")):
                canvas.fill_text(
                    text = line, x = x0, y = y0 + fontsize * i, 
                    max_width = max_width
                )


    @classmethod
    def _draw_loading(cls, canvas, message = "Loading...", wrap = True, fontsize = 32):
        """Clears the canvas and draws the loading screen."""
        cls._draw_text_canvas(
            canvas   = canvas, 
            message  = message, 
            wrap     = wrap, 
            fontsize = fontsize
        )

    # Message Canvas Methods ---------------------------------------------------

    def write_message(self, message, wrap = True, fontsize = 32):
        """Writes a message on the message canvas."""
        self._draw_text_canvas(
            canvas   = self.message_canvas, 
            message  = message, 
            wrap     = wrap, 
            fontsize = fontsize
        )
  

    def clear_message(self):
        """Clears the current message canvas."""
        self.message_canvas.clear()

    # Update State Methods -----------------------------------------------------

    @staticmethod
    def empty_point_matrix():
        return np.zeros((0, 2), dtype = float)


    def calc_fixed_point(self, annotation, target_annotations, fixed_point):
        """Calculates the fixed head or tail point for the given annotation."""
        if fixed_point not in ("fixed_head", "fixed_tail"):
            raise ValueError(f"Invalid fixed point: {fixed_point}")

        # Get the fixed head or tail attribute for the given annotation.
        fixed_point = getattr(self.annot_cfg[annotation], fixed_point)

        # If there is a fixed head, we need to calculate it using the provided function.
        if fixed_point is not None:
            try:
                fixed_point = fixed_point["calculate"](target_annotations)
                fixed_point = fixed_point.reshape(1, 2)
            except Exception:
                fixed_point = None
        
        # Return the fixed point (None or coordinates of the fixed point).
        return fixed_point


    @staticmethod
    def _init_editable(x = None):
        """Initializes the editable points for the given annotation."""
        if x is None: return np.zeros((0,), dtype = int)
        return np.array([x], dtype = int)
    

    def _calc_editable(self):
        """Calculates the editable points for the active annotation."""
        # Get the points, fixed head, and fixed tail for the active annotation
        points = self.annotations[self.active]
        fixed_head = self.fixed_heads[self.active]
        fixed_tail = self.fixed_tails[self.active]

        # Determine which points are fixed by comparing them to the fixed head and tail.
        fixed_head = np.all(points == fixed_head, axis = 1)
        fixed_tail = np.all(points == fixed_tail, axis = 1)
        fixed_index = np.logical_or(fixed_head, fixed_tail)

        # Return the indices of the editable points (i.e., non-fixed points).
        return np.where(~fixed_index)[0]


    def update_state(self, target_id, annotation, target_annotations):
        """Updates the state to reflect the given target and annotation."""
    
        # If neither the target nor the annotation is changing, we can skip the update.
        if self.target == target_id and self.active == annotation: return

        # Store the previous state.
        prev_target     = self.target
        prev_annotation = self.active

        # Update the target, active annotation, and annotations.
        self.target      = target_id
        self.active      = annotation
        self.annotations = target_annotations

        # Update the image data, grid shape, and figure limits from the state.
        image_data, grid_shape, meta_data = self.state.grid(
            self.target, self.active)
        self.image = ipw.Image(value = image_data, format = "png")
        self.grid       = self.annot_cfg.figure_grid[self.active]
        self.grid_shape = grid_shape
        self.xlim = meta_data["xlim"]
        self.ylim = meta_data["ylim"]

        # If the target is changing, we need to reset the fixed heads and tails, 
        # since they are target specific. Recalculating everything.
        if self.fixed_heads == {} or self.fixed_tails == {} or \
            prev_target != self.target: 
            self.fixed_heads = {}
            self.fixed_tails = {}
            recalc_fixed     = list(self.annotations.keys())
        # If the annotation is changing, we need to recalculate the fixed heads
        # tails for dependencies of the previous annotation.
        else:
            prev_deps = self.annot_cfg.fixed_dependencies.get(prev_annotation, [])
            recalc_fixed = { self.active, *prev_deps}
            
        # Recalculate the fixed head and tails of the given fixed annotations.
        for annotation in recalc_fixed:
            self.fixed_heads[annotation] = self.calc_fixed_point(
                annotation, self.annotations, "fixed_head")
            self.fixed_tails[annotation] = self.calc_fixed_point(
                annotation, self.annotations, "fixed_tail")
        
        # Get the points and annotation type for the active annotation.
        points = self.annotations[self.active]
        atype  = self.annot_cfg.types[self.active]

        # If there are no points for the current annotation, initialize.
        if points is None or points.shape[0] == 0:
            points = self.empty_point_matrix()

        # Determine the editable points.
        if atype == "point":
            # Points annotations either have no point or exactly one point.
            if points.shape[0] == 0:
                self.editable = self._init_editable()
            else:
                self.editable = self._init_editable(0)
        else: # atype in ( "contour", "boundary" )
            # If points is empty, update the annotations with the fixed points. 
            # Annotations should be saved WITH their fixed heads and tails.
            if points.shape[0] == 0:
                if self.fixed_heads[self.active] is not None:
                    points = np.vstack([self.fixed_heads[self.active], points])
                if self.fixed_tails[self.active] is not None:
                    points = np.vstack([points, self.fixed_tails[self.active]])
                    
                # Update the annotation with the new points, if necessary.
                self.annotations[self.active] = points

            # Calculate the editable points (non-fixed points)
            self.editable = self._calc_editable()
    
        # If there are no editable points, we set the cursor to None.
        # Otherwise, we set the cursor to the last editable point.
        if self.editable.shape[0] == 0:
            self.cursor = None
        else:
            self.cursor = self.editable[-1]

    # Canvas Resizing Method ---------------------------------------------------

    def resize_canvas(self, new_figure_size = None):
        """Resizes the figure canvas so that images appear at the given scale.

        `figure_panel.resize_canvas(new_figure_size)` results in the canvas being
        resized to match the new figure size. Note that this does not resize the
        canvas to have a width of `new_figure_size` but rather resizes it so that each
        image in the grid has a width of `new_figure_size`.

        The `resize_canvas` method triggers a redraw because the resizing of the
        canvas clears it.
        """
        # If there is no new_figure_size give, we just use the current figure size.
        if new_figure_size is None:
            new_figure_size = self.state.figure_size()

        # Set the new figure size.
        self.figure_size = np.array([new_figure_size, new_figure_size])

        # The canvas size is a product of the figure size and the grid shape.
        self.canvas_size = self.figure_size * np.array(self.grid_shape)
        canvas_width, canvas_height = self.canvas_size.astype(int)

        # First resize the canvas (this clears it).
        self.multicanvas.width  = canvas_width
        self.multicanvas.height = canvas_height

        # Then we also resize the layout component.
        self.multicanvas.layout.width  = f"{canvas_width}px"
        self.multicanvas.layout.height = f"{canvas_height}px"

        # Finally, because the canvas was cleared upon resize, we redraw it.
        self.redraw_canvas()
    
    # Redraw Mulicanvas Method -------------------------------------------------

    def redraw_canvas(
            self, image = None, grid_shape = None, xlim = None, ylim = None,
            redraw_image = True, redraw_annotations = True
        ):
        """Redraws the entire canvas.

        `figure_panel.redraw_canvas()` redraws the canvas as-is.

        `figure_panel.redraw_canvas(new_image)` redraws the canvas with the
        new image; this requires that the grid has not changed.

        `figre_panel.redraw_canvas(new_image, new_grid_shape)` redraws the
        canvas with the given new image and new grid shape.

        The optional arguments `redraw_image` and `redraw_annotations` both
        default to `True`. They can be set to `False` to skip the redrawing of
        one or the other layer of the canvas.
        """
        # If no image give, redraw the current image.
        if image is None:
            image = self.image
        else: # If an image is given, we update the current image.
            self.image = image

        # Update the xlim and ylim if given.
        if xlim is not None:
            self.xlim = xlim
        if ylim is not None:
            self.ylim = ylim

        # If no grid shape is given, redraw with the current grid shape.
        if grid_shape is None:
            grid_shape = self.grid_shape
        elif grid_shape != self.grid_shape:
            # If grid shape is given and different from current grid shape, we
            # update and resize the canvas, which will trigger another redraw, 
            # so we return here to avoid doing a redudant redraw. 
            self.grid_shape = grid_shape
            self.resize_canvas()
            return

        # Redraw the loading canvas (assuming one was given).
        if redraw_image or redraw_annotations:
            self.loading_canvas.restore()
        
        # Redraw the image and annotations, if necessary.
        with ipc.hold_canvas():
            if redraw_image: 
                self.redraw_image()
            if redraw_annotations: 
                self.redraw_annotations()
                self._increment_annotation_change()

    # Canvas to Figure Coordinate Conversion Method ----------------------------

    def canvas_to_figure(self, points):
        """Converts the `N x 2` matrix of canvas points into figure coordinates."""
        # Check the shape of the input and convert it into an `N x 2` matrix if necessary.
        points = np.asarray(points)
        if len(points.shape) == 1:
            return self.canvas_to_figure([points])[0]
        
        # First off, we want to apply the grid mod to make sure that any points 
        # that are outside the figure limits get wrapped around to the location.
        (figure_width, figure_height) = self.figure_size
        points = points % [ figure_width, figure_height ]
        
        # Get the figure limits.
        xlim = (0, figure_width) if self.xlim is None else self.xlim
        ylim = (0, figure_height) if self.ylim is None else self.ylim
        
        # We need to invert the y axis.
        points[:,1] = figure_height - points[:,1]

        # Now, make the conversion.
        points *= [(xlim[1] - xlim[0]) / figure_width,
                   (ylim[1] - ylim[0]) / figure_height]
        points += [xlim[0], ylim[0]]

        # Return the converted points.
        return points

        
    def figure_to_canvas(self, points):
        """Converts the `N x 2` matrix of figure points into canvas coordinates."""
        # Check the shape of the input and convert it into an `N x 2` matrix if necessary.
        points = np.asarray(points)
        if len(points.shape) == 1:
            return self.figure_to_canvas([points])[0]
        # Get the figure limits.
        (figure_width, figure_height) = self.figure_size
        xlim = (0, figure_width) if self.xlim is None else self.xlim
        ylim = (0, figure_height) if self.ylim is None else self.ylim

        # First, make the basic conversion.
        points  = points - [xlim[0], ylim[0]]
        points *= [figure_width / (xlim[1] - xlim[0]),
                   figure_height / (ylim[1] - ylim[0])]
        
        # Then invert the y-axis
        points[:,1] = figure_height - points[:,1]

        # And build up the point matrices for each grid element.
        (n_rows, n_cols) = self.grid_shape # grid shape
        return [
            points + [ii * figure_width, jj * figure_height]
            for ii in np.arange(n_cols)
            for jj in np.arange(n_rows)
            if self.grid[jj][ii] is not None
        ]


    # Mouse Event Handler Methods ----------------------------------------------

    @staticmethod
    def _to_point_matrix(x, y = None):
        x = np.asarray(x) if y is None else np.array([[x, y]])
        if x.shape == (2,):
            x = x[None,:]
        elif x.shape != (1,2):
            raise ValueError(f"Bad point shape: {x.shape}")
        return x


    def _recalculate_deps(self, annotation):
        """Recalculates the dependent annotations for the given annotation."""
        # Get the dependent annotations for the given annotation.
        fixed_deps = self.annot_cfg.fixed_dependencies[annotation]

        # If there are no dependencies, we can skip.
        if len(fixed_deps) == 0: return 

        # We need to recalculate each of the dependent annotations using their
        # provided functions and update them in the state.
        for fd in fixed_deps: 
            # Get the current points for the dependent annotation.
            points = self.annotations[fd]

            # If there are no points, we can skip the recalculation.
            if points is None or points.shape[0] == 0: continue

            # Recalculate and update the fixed head for the dependent annotation.        
            fixed_head = self.calc_fixed_point(fd, self.annotations, "fixed_head")
            if fixed_head is not None:
                points[0,:] = fixed_head

            # Recalculate and update the fixed tail for the dependent annotation.        
            fixed_tail = self.calc_fixed_point(fd, self.annotations, "fixed_tail")
            if fixed_tail is not None:
                points[-1,:] = fixed_tail

            # Update the annotation with the new points.
            self.annotations[fd] = points
    
    
    def _increment_annotation_change(self):
        """Increments the annotation change traitlet after redraw triggers."""
        self._annotation_change += 1


    def push_point(self, x, y = None, redraw = True):
        """Push the given point onto the path at the cursor end.

        The point may be given as `x, y` or as a vector or 1 x 2 matrix. The
        point is added to the head or the tail depending on the cursor.
        """
        # We can only push points if there is an active annotation.
        if self.active is None: return None

        # First convert the input into a point matrix, must be N x 2.
        new_point = FigurePanel._to_point_matrix(x, y)

        # Get the current points for this annotation. If None, initialize empty.
        points = self.annotations[self.active]
        if points is None: points = self.empty_point_matrix()

        # Get the annotation type for this annotation.
        atype = self.annot_cfg.types[self.active]
        
        # Depending on the annotation type, we add the newest point to the
        # annotation in different ways.
        if atype == "point":
            # For a point annotation, replace the current point with the new point.
            points        = new_point
            self.editable = self._init_editable(0)
            self.cursor   = 0

        else: # atype in ( "contour", "boundary" )
            # If there are no points, we just add the new point.
            if points.shape[0] == 0:
                self.editable = self._init_editable(0)
                self.cursor   = 0

            # If there are no editable points, we add the new point to the head
            # or tail depending on which one is fixed.                
            elif self.editable.shape[0] == 0:
                if self.fixed_heads[self.active] is not None:
                    self.editable = self._init_editable(1)
                elif self.fixed_tails[self.active] is not None:
                    self.editable = self._init_editable(0)
                self.cursor = self.editable[0]   

            # If there are editable points, we add the new point after the 
            # current cursor position and move the cursor to the new point.
            else: 
                # Because we are inserting a point, all the editable points 
                # after the cursor need to be shifted by one index.
                self.editable[self.editable > self.cursor] += 1

                # We add the new cursor position to the editable points.
                self.editable = np.sort(np.append(self.editable, self.cursor + 1))
                
                # Finally, we increment the cursor to move it to the next position.
                self.cursor += 1

            # Insert the new point at the cursor position.
            points = np.insert(points, self.cursor, new_point, axis = 0)
 
        # Update the annotation with the new points.
        self.annotations[self.active] = points

        # Update dependent annotations, if this active annotation has them.
        fixed_deps = self.annot_cfg.fixed_dependencies[self.active]
        has_deps   = len(fixed_deps) > 0
        if has_deps: self._recalculate_deps(self.active)

        # Redraw the annotations.
        if redraw:
            with ipc.hold_canvas():
                self.redraw_annotations(background = has_deps)
            self._increment_annotation_change()


    def _push_impoint(self, x, y = None, redraw = True):
        """Push the given canvas point onto the selected annotation.

        The point may be given as `x, y` or as a vector or 1 x 2 matrix. Canvas
        points are always converted into figure points before being appended to
        the annotation. The point is added to the head or the tail depending on
        the cursor.
        """
        # First convert the input into a point matrix, must be N x 2.
        x = FigurePanel._to_point_matrix(x, y)

        # Convert to a figure point.
        x = self.canvas_to_figure(x)

        # And then push it onto the annotation.
        return self.push_point(x, redraw = redraw)
    

    def on_mouse_click(self, x, y):
        """This method is called when the mouse is clicked on the canvas."""        
        # If the figure is locked, we do not allow events. Skip.
        if self.state.locked: return

        # Add to the current contour.
        self._push_impoint(x, y)
    
    # Key Press Event Handler Methods ------------------------------------------

    def toggle_cursor(self):
        """Toggles the cursor position of the active annotation."""
        # If the figure is locked, we do not allow events. Skip.
        if self.state.locked: return

        # Extract current annotation type.
        atype  = self.annot_cfg.types[self.active]

        # For a point annotation, there is only one point. Toggling the 
        # cursor position does not do anything, so we can skip it.
        if atype == "point": return

        # If there are less than two editable points, we cannot toggle the cursor.
        if self.editable.shape[0] < 2: return

        # For contour or boundary annotations, we toggle the cursor position by 
        # moving it to the next editable point in the annotation.
        if atype in ( "contour", "boundary" ):
            # Get the index of the current cursor position in the editable points.
            current_index = np.where(self.editable == self.cursor)[0][0]

            # Calculate the index of the next editable point with wraparound.
            next_index = np.mod(current_index + 1, self.editable.shape[0])

            # Update the cursor to the next editable point.
            self.cursor = self.editable[next_index]

        # Redraw the annotations to show the new cursor position.
        with ipc.hold_canvas():
            self.redraw_annotations(background = False)

    
    def pop_point(self, redraw = True):
        """Removes the point at the current cursor position of the active annotation."""
        # We can only push points if there is an active annotation.
        if self.active is None: return None

        # Get the current annotation and annotation type.
        points = self.annotations[self.active]
        atype  = self.annot_cfg.types[self.active]

        # If there are no points, we cannot delete anything. Skip.
        if points is None or points.shape[0] == 0 or \
            self.editable.shape[0] == 0: return
        
        # Check if there are any LIVE dependencies on this annotation. If so, 
        # we cannot delete the last point of this annotation because the 
        # dependent annotations rely on it. 
        fixed_deps = self.annot_cfg.fixed_dependencies[self.active]
        has_deps   = len(fixed_deps) > 0
        if has_deps and self.editable.shape[0] == 1:
            # Determine the number of fixed points for each dependent 
            # annotation. This number is the minimum number of points that the 
            # annotation must have be considered LIVE.
            n_fixed = [ len(self.annot_cfg.fixed_points[fd]) for fd in fixed_deps ]

            live_deps = [
                fd for fd, n in zip(fixed_deps, n_fixed) 
                if self.annotations[fd] is not None
                and self.annotations[fd].shape[0] > n
            ]
        
            if live_deps:
                # Write a warning message to the user about live dependencies. 
                self.write_message(
                    f"Cannot delete: '{self.active}'. It is required by "
                    f"'{', '.join(live_deps)}'. Clear those annotations first."
                )
                # Clear the message after 3 seconds. 
                threading.Timer(3.0, self.clear_message).start()
                return
        
        # If there are points, we delete based on annotation type.
        if atype == "point":
            # For a point annotation, we delete the single point.
            points        = self.empty_point_matrix()
            self.editable = self._init_editable()
            self.cursor   = None
        else: # atype in ( "contour", "boundary" )
            # If there are points to delete, delete at current position.
            points = np.delete(points, self.cursor, axis = 0)

            # Remove the current cursor from the editable points.
            self.editable = self.editable[self.editable != self.cursor]
            if self.editable.shape[0] == 0:
                self.cursor = None
            else:
                # Removing an index causes all the indices larger than the 
                # current position to shift down by one, so we need to decrement
                # the editable points.
                self.editable[self.editable > self.cursor] -= 1

                # When the cursor is at the head of the editable points, we do
                # not need to decrement the cursor because it will just move 
                # down with the shift of the points. However, if the cursor is
                # anywhere else, we need to decrement the cursor.
                if self.cursor != self.editable[0]: 
                    self.cursor -= 1

        # Update the annotation with the new points.
        self.annotations[self.active] = points

        # Update dependent annotations, if this active annotation has them.
        if has_deps: self._recalculate_deps(self.active)

        # Redraw the annotations.
        if redraw:
            with ipc.hold_canvas():
                self.redraw_annotations(background = has_deps)
            self._increment_annotation_change()
   

    def on_key_press(self, key, shift_down, ctrl_down, meta_down):
        """This method a key is pressed."""
        # If the figure is locked, we do not allow events. Skip.
        if self.state.locked: return
        
        # Handle the key press.
        key = key.lower()
        if key == "tab":
            # Toggle the cursor (active) position.
            self.toggle_cursor()
        elif key == "backspace":
            # Delete current cursor (active) point.
            self.pop_point()
        else:
            pass
