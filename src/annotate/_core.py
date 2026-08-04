# -*- coding: utf-8 -*-
################################################################################
# annotate/_core.py

"""Core implementation code for the annotation tool's interface.

This file primarily contains code for managing the widget and window state of
the control panel; the canvas and figure code is largely handled by the
FigurePanel widget in the _figure.py file.
"""


# Imports ######################################################################

import os
import re
import json
import yaml
import numpy as np
import pandas as pd
import os.path as op
import matplotlib as mpl
import ipywidgets as ipw
import imageio.v3 as iio
from warnings import warn
import matplotlib.pyplot as plt

from ._util    import (ldict, delay)
from ._config  import Config
from ._control import ControlPanel
from ._figure  import FigurePanel


# The State Manager ############################################################

class NoOpContext:
    def __enter__(self): pass
    def __exit__(self, type, value, traceback): pass


class AnnotationState:
    """The manager of the state of the annotation and the annotation tool.

    The `AnnotationState` class manages the state of the annotation tool. This
    state includes the cache, the user preferences (style settings), and the
    saved annotations.
    """
    
    DEFAULT_STYLE = {
        "color"      : "black",
        "linestyle"  : "solid",
        "linewidth"  : 1,
        "markersize" : 1,
        "visible"    : True
    }

    STYLE_KEYS = tuple(DEFAULT_STYLE.keys())

    __slots__ = (
        "config", "cache_path", "save_path", "git_path", "username",
        "annotations", "preferences", "loading_context", "save_hooks", 
        "locked"
    )
    
    def __init__(
            self,
            config_path     = "/config/config.yaml",
            cache_path      = "/cache",
            save_path       = "/save",
            git_path        = "/git",
            username        = None,
            loading_context = None,
        ):

        # Store the configuration and paths.
        self.config     = Config(config_path)
        self.cache_path = cache_path
        self.save_path  = save_path
        self.git_path   = git_path
        self.save_hooks = None

        # We add the git username to the save path if needed here.
        if username is None:
            (username, git_reponame) = self.gitdata
        if not isinstance(username, str):
            raise RuntimeError("username must be a string or None")

        # Build up the save path.
        self.username = username
        if username != "": # if username, add as subdirectory of save path
            self.save_path = op.join(save_path, username)
        if not op.isdir(self.save_path):
            os.makedirs(self.save_path, mode = 0o755)

        # Use our loading control if we have one.
        if loading_context is None:
            loading_context = NoOpContext()
        self.loading_context = loading_context

        # (Lazily) load the annotations.
        self.annotations = self.load_annotations()

        # And (lazily) load the preferences.
        self.preferences = self.load_preferences()

        # Declare the locked state of the annotation tool. When locked, the user
        # cannot interact with the figure panel and some control panel options 
        # are disabled. This is used when there is an error with the current
        # selection that prevents the figure from being properly displayed.
        self.locked = False
        
    # Git Methods --------------------------------------------------------------

    @property
    def gitdata(self):
        """Reads and returns the repo username and the repo name."""
        # If we were not given a git path, we return standard nothings.
        if self.git_path is None:
            return ( "", "" )
        try:
            # For some reason, it seems that sometimes docker does not fully
            # mount the directory until we've attempted to list its contents.
            with os.popen(f"ls {self.git_path}") as f: f.read()
            # Having performed an ls, go ahead and check git's opinion about the
            # origin with git config command line calls.
            cmd  = f"cd {self.git_path}"
            cmd += f" && git config --global --add safe.directory {self.git_path}"
            cmd +=  " && git config --get remote.origin.url"
            with os.popen(cmd) as p:
                repo_url = p.read().strip()
            repo_split = repo_url.split("/")
            repo_name = repo_split.pop()
            while repo_name == "":
                repo_name = repo_split.pop()
            repo_user = repo_split.pop()
            s1 = repo_user.split("/")[-1]
            s2 = repo_user.split(":")[-1]
            repo_user = s1 if len(s1) < len(s2) else s2
            return ( repo_user, repo_name )
        except Exception as e:
            # If there was an error, we just warn and return nothings.
            warn(f"Error finding gitdata: {e}")
            return ( "", "" )
    
    # Pathing Methods ----------------------------------------------------------

    def _target_path(self, target):
        """Returns the relative path for a target."""
        if isinstance(target, tuple):
            path = target
        else:
            path = [target[k] for k in self.config.targets.concrete_keys]
        return op.join(*path)
    
    
    def target_figure_path(self, target, figure = None, ensure = True):
        """Returns the cache path for a target's figures."""
        path = self._target_path(target)
        path = op.join(self.cache_path, "figures", path)
        if ensure and not op.isdir(path):
            os.makedirs(path, mode = 0o755)
        if figure is not None:
            path = op.join(path, f"{figure}.png")
        return path
    
    
    def target_grid_path(self, target, annotation = None, ensure = True):
        """Returns the cache path for a target's grids."""
        path = self._target_path(target)
        path = op.join(self.cache_path, "grids", path)
        if ensure and not op.isdir(path):
            os.makedirs(path, mode = 0o755)
        if annotation is not None:
            path = op.join(path, f"{annotation}.png")
        return path
    
    
    def target_save_path(self, target, annotation = None, ensure = True):
        """Returns the save path for a target's annotation data."""
        path = self._target_path(target)
        path = op.join(self.save_path, path)
        if ensure and not op.isdir(path):
            os.makedirs(path, mode = 0o755)
        if annotation is not None:
            path = op.join(path, f"{annotation}.tsv")
        return path
    
    # Figure/Grid Methods ------------------------------------------------------
    
    def _generate_figure(self, target_id, figure_name):
        """Generates a single figure for the given target and figure name."""
        # Get the current target.
        target = self.config.targets[target_id]
        
        # Prepare the image and meta data file paths.
        impath = self.target_figure_path(target, figure_name)
        mdpath = re.sub(".png$", ".json", impath)
        
        # Get the display settings and figure function.
        figsize, dpi = self.config.display.figsize, self.config.display.dpi
        figure_fn = self.config.figures[figure_name]

        # Run the function from the config that draws the figure.
        (fig, ax) = plt.subplots(1, 1, figsize = figsize, dpi = dpi)
        meta_data = {} # initalize, can be populated by figure function
        figure_fn(target, figure_name, fig, ax, figsize, dpi, meta_data)
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        ax.axis("off")

        # Save the figure out as a png file.
        plt.savefig(impath, bbox_inches = None)
        
        # We also need a companion meta-data file.
        if "xlim" not in meta_data: meta_data["xlim"] = ax.get_xlim()
        if "ylim" not in meta_data: meta_data["ylim"] = ax.get_ylim()

        # Save the meta data as a json file.
        jscode = json.dumps(meta_data)
        with open(mdpath, "wt") as f:
            f.write(jscode)

        # We can close the figure now as well.
        plt.close(fig)
    

    def figure(self, target_id, figure_name):
        """Returns the image and metadata for the given target and figure name.
        
        The return value is `(image_data, meta_data)` where the `image_data` is
        a numpy array of the image data, and the `meta_data` is a `dict`.
        """
        if figure_name is None:
            # This is a request for an empty image.
            image_size = self.config.display.image_size
            return ( np.ones(image_size + (4,), dtype = np.uint8) * 255, None)
        
        # Prepare the image and meta data file paths.
        impath = self.target_figure_path(target_id, figure_name)
        mdpath = re.sub(".png$", ".json", impath)   
        
        # If the files does not already exist, we generate them first.
        if not op.isfile(impath) or not op.isfile(mdpath):
            with self.loading_context:
                self._generate_figure(target_id, figure_name)
        
        # Now read the figure image data and meta data.
        image_data = iio.imread(impath)
        with open(mdpath, "rt") as f:
            meta_data = json.load(f)

        # And return the image data and meta data.
        return ( image_data, meta_data )


    def _generate_grid(self, target_id, annotation):
        """Generates a single figure grid for an annotation."""
        # Prepare the image and meta data file paths.
        impath = self.target_grid_path(target_id, annotation)
        mdpath = re.sub(".png$", ".json", impath)

        # Get the annotation information for this annotation.
        annotation_info = self.config.annotations[annotation]
        
        # Get the figure image/meta data for the entire figure grid
        figure_data = [
            [ self.figure(target_id, figure_name) for figure_name in row ]
            for row in annotation_info.figure_grid
        ]
        
        # Make sure the figure xlim and ylim meta-data all match!
        meta_data = [ md for row in figure_data for (_, md) in row ]
        meta_data0 = meta_data[0] # we use this as the reference meta data
        for md in meta_data: # for each figure in the grid
            if md is not None: # skip the empty figures
                if meta_data0["xlim"] != md["xlim"]:
                    raise RuntimeError(f"Not all figures have the same `xlim` "
                                    f"for annotation: {annotation}")
                if meta_data0["ylim"] != md["ylim"]:
                    raise RuntimeError(f"Not all figures have the same `ylim` "
                                    f"for annotation: {annotation}")
                
        # Concatenate the figures to make a single grid image.
        grid = np.concatenate([
            np.concatenate([fig for (fig, _) in row], axis = 1)
            for row in figure_data], axis = 0
        )
        
        # Save it out as a png file.
        iio.imwrite(impath, grid)

        # And save out the meta-data.
        jscode = json.dumps(meta_data0)
        with open(mdpath, "wt") as f:
            f.write(jscode)


    def grid(self, target_id, annotation):
        """Returns the grid of figures for the given target and annotation.

        The return value is `(image_data, grid_shape, meta_data)` where the
        `image_data` is the raw bytes of the file, `grid_shape` is a tuple of
        the `(row_count, column_count)` of the grid, and the `meta_data` is a
        `dict`.
        """
        # Prepare the image and meta data file paths.
        impath = self.target_grid_path(target_id, annotation)
        mdpath = re.sub(".png$", ".json", impath)

        # Get the annotation information for this annotation.
        annotation_info = self.config.annotations[annotation]
        grid_shape = np.shape(annotation_info.figure_grid) 

        # If the files aren't here already, we generate them first.
        if not op.isfile(impath) or not op.isfile(mdpath):
            with self.loading_context:
                self._generate_grid(target_id, annotation)
        
        # Read in image data. 
        with open(impath, "rb") as f:
            image_data = f.read()

        # Read in meta data.
        with open(mdpath, "rt") as f:
            meta_data = json.load(f)
        
        # And return them.
        return ( image_data, grid_shape, meta_data )

    # Annotation Methods -------------------------------------------------------

    def load_target_annotation(self, target_id, annotation):
        """Loads a single annotation from the save path for a given target."""
        # Get the path for this annotation.
        tsv_file = self.target_save_path(target_id, annotation)

        # If there is no file or the file is empty, we return an empty matrix of points.
        if not op.isfile(tsv_file) or op.getsize(tsv_file) == 0:
            return np.zeros((0, 2), dtype = float)

        # Read in the coordinates using pandas (tab separated, no header).
        coords = pd.read_csv(tsv_file, sep = "\t", header = None).values

        # The TSV file must contain an N x 2 matrix of values!
        if len(coords.shape) != 2 or coords.shape[1] != 2:
            raise RuntimeError(
                f"File '{tsv_file}' for annotation '{annotation}' and "
                f"target '{target_id}' has invalid shape: {coords.shape}"
            )

        # Return the coordinates.
        return coords
    
    
    def load_target_annotations(self, target_id):
        """Loads (lazily) the annotations for the current tool user for a single target"""
        target_annotations = ldict() # initialize
        for annotation in self.config.annotations.keys():
            target_annotations[annotation] = delay(
                self.load_target_annotation, target_id, annotation)
        return target_annotations

    
    def load_annotations(self):
        """Loads (lazily) the annotations for the current tool user from the save path."""
        return ldict({
            target_id: delay(self.load_target_annotations, target_id)
                for target_id in self.config.targets.keys()
            })


    def save_target_annotations(self, target_id):
        """Saves the annotations for the current tool user for a single target"""
        # Get the target's annotations.
        target_annotations = self.annotations[target_id]

        for annotation_name in target_annotations.keys(): 
            # Skip anything lazy. We never want to save anything that's still
            # lazy because that means that the original file hasn't been read in
            # (and thus can't have any updates).
            if target_annotations.is_lazy(annotation_name): continue
            
            # Get this annotation's coordinates.
            coords = np.asarray(target_annotations.get(annotation_name))

            # Make sure they are the right shape.
            if len(coords.shape) != 2 or coords.shape[1] != 2:
                raise RuntimeError(
                    f"Annotation '{annotation_name}' for target "
                    f"{target_id} has invalid shape: {coords.shape}"
                )
            
            # If they're empty, no need to save them.
            tsv_file = self.target_save_path(target_id, annotation_name)
            if coords.shape[0] == 0: 
                # delete the file if it exists instead.
                if op.isfile(tsv_file): os.remove(tsv_file)
                continue

            # Save them using pandas.
            df = pd.DataFrame(coords)
            df.to_csv(tsv_file, index = False, header = None, sep = "\t")
    
    
    def save_annotations(self):
        """Saves the annotations for a given target."""
        annotations = self.annotations
        for target_id in annotations.keys():
            # Skip lazy keys; these targets have not even been loaded yet.
            if not annotations.is_lazy(target_id):
                self.save_target_annotations(target_id)
    
    # Preferences Methods ------------------------------------------------------

    def load_preferences(self):
        """Loads the preferences from the save directory and returns them.

        If no preferences file is found, an empty dictionary is returned.
        """
        preferences_yaml = op.join(self.save_path, ".annot-prefs.yaml")
        if not op.isfile(preferences_yaml):
            # If there is no preferences file, initailize the preferences
            preferences = { "style": {}, "figure_size": 256 } 

            # For each annotation, set the default style dictionary.
            # DEFAULT_STYLE << config.display.default_style
            styledict = AnnotationState.DEFAULT_STYLE.copy()
            styledict = { **styledict, **self.config.display.default_style }
            for annotation in self.config.annotations.keys():
                preferences["style"][annotation] = styledict.copy()
            
            # Set the annotation for the active style as None.
            # DEFAULT_STYLE << config.display.default_style << config.display.active_style
            styledict = { **styledict, **self.config.display.active_style }
            preferences["style"][None] = styledict.copy()

            # Return the preferences.
            return preferences
        with open(preferences_yaml, "rt") as f:
            return yaml.safe_load(f)
    
    
    def save_preferences(self):
        """Saves the preferences to the save directory."""
        preferences_yaml = op.join(self.save_path, ".annot-prefs.yaml")
        with open(preferences_yaml, "wt") as f:
            yaml.dump(self.preferences, f)
    
    # Figure Size Methods ------------------------------------------------------

    def figure_size(self, new_figure_size = None):
        """Returns the figure size from the user's preferences.

        `state.figure_size()` returns the current figure size.

        `state.figure_size(new_figure_size)` updates the current figure size.
        """
        if new_figure_size is None:
            # Just return the current figure size, or the default if it is not set.
            return self.preferences.get("figure_size", 256)
        else:
            # Update the figure size in the preferences, and return the new value.
            self.preferences["figure_size"] = new_figure_size
            return new_figure_size

    # Style Methods ------------------------------------------------------------

    @classmethod
    def fix_style(cls, style_dict):
        """Ensures that the given dictionary is valid as a style dictionary."""
        # Check that all the keys are valid style keys.
        for key in style_dict.keys():
            if key not in AnnotationState.STYLE_KEYS:
                raise RuntimeError(f"Invalid style key: {key}")
            
        # Check that the linewidth is a valid number.
        if "linewidth" in style_dict:
            linewidth = style_dict["linewidth"]
            if linewidth < 0 or linewidth > 20:
                raise RuntimeError(f"Invalid linewidth: {linewidth}")
        
        # Check that the linestyle is valid.
        if "linestyle" in style_dict:
            linestyle = style_dict["linestyle"]
            if linestyle not in ("solid", "dashed", "dot-dashed", "dotted"):
                raise RuntimeError(f"Invalid linestyle: {linestyle}")
            
        # Check that the color is valid.
        if "color" in style_dict:
            color = style_dict["color"]
            try: color = mpl.colors.to_hex(color)
            except Exception as e: 
                raise RuntimeError(f"Invalid color: {color}") from e
            style_dict["color"] = color # store as hex, if valid

        # Check that the markersize is a valid number.
        if "markersize" in style_dict:
            markersize = style_dict["markersize"]
            if markersize < 0 or markersize > 20:
                raise RuntimeError(f"Invalid markersize: {markersize}")
        
        # Check that the visible is a boolean.
        if "visible" in style_dict:
            visible = style_dict["visible"]
            if not isinstance(visible, bool):
                raise RuntimeError(f"Invalid visible: {visible}")
        
        # Return the style dictionary, if valid.
        return style_dict
    

    def style(self, annotation, *args):
        """Returns the style dict of the given annotation.

        `state.style(annot)` returns the current styledict for the
        annotation named `annot`. This style dictionary is always fully reified
        with all style keys.

        `state.style(annot, new_styledict)` updates the current styledict
        to have the contents of `new_styledict` then returns the new value.

        `state.style(annot, key, value)` is equivalent to
        `state.style(annot, { key : value })`.
        
        The styledict contains the keys `"linewidth"`, `"linestyle"`,
        `"markersize"`, `"color"`, and `"visible"`.
        """
        # Check the annotation name is valid.
        if annotation is not None and annotation not in self.config.annotations:
            raise RuntimeError(f"Invalid annotation name: {annotation}")

        # Check the number of argumments 
        nargs = len(args)
        if nargs > 1 and nargs % 2 != 0:
            raise RuntimeError("Invalid number of arguments given to styledict.")
            
        # In all cases, we start by calculating our own styledict.
        # See if there is a dictionary in the preferences already.
        preferences = self.preferences["style"]
        if nargs == 0:
            # We're just returning the current annotation styledict.
            new_styledict = preferences.get(annotation, {})
        elif nargs == 1:
            # We're creating a new styledict based on the provided dict.
            new_styledict = self.fix_style(args[0])
        else:
            # We're creating a new styledict based on the provided key-value pairs.
            new_styledict = self.fix_style(
                { key: value for (key, value) in zip(args[0::2], args[1::2])})
            
        # Update user's preferences with the new styledict for this annotation.
        preferences[annotation] = { **preferences[annotation], **new_styledict }
        self.preferences["style"] = preferences

        # And return the updated styledict for the queried annotation.
        return preferences[annotation]

# The Annotation Tool ##########################################################

class AnnotationTool(ipw.HBox):
    """The core annotation tool for the `cortex-annotate` project.

    The `AnnotationTool` type handles the annotation of the cortical surface
    images for the `cortex-annotate` project.
    """

    def __init__(
            self,
            config_path = "/config/config.yaml",
            cache_path  = "/cache",
            save_path   = "/save",
            git_path    = "/git",
            username    = None,
            control_panel_background_color = "#f0f0f0",
            button_color = "#e0e0e0",
        ):        
        """Initializes the annotation tool."""
        
        # Store the state.
        self.state = AnnotationState(
            config_path = config_path,
            cache_path  = cache_path,
            save_path   = save_path,
            git_path    = git_path,
            username    = username
        )

        # Pull out the annotation config for easy access.
        self.annot_cfg = self.state.config.annotations

        # Make the control panel.
        self.control_panel = ControlPanel(
            state             = self.state,
            background_color  = control_panel_background_color,
            button_color      = button_color,
        )
        
        # Make the figure panel.
        self.figure_panel = FigurePanel(self.state)
        
        # Pass the loading context over to the state.
        self.state.loading_context = self.figure_panel.loading_context

        # Go ahead and initialize the HBox component.
        super().__init__([self.control_panel, self.figure_panel])

        # Give the figure the initial image to plot.
        with self.state.loading_context:
            self.refresh_figure()

        # And a listener for the selection change.
        self.control_panel.observe_selection(self.on_selection_change)

        # Add a listener for the figure size change.
        self.control_panel.observe_figure_size(self.on_figure_size_change)

        # And a listener for the style change.
        self.control_panel.observe_style(self.on_style_change)

        # Add a listener for the clear all button.
        self.control_panel.observe_clear(self.on_clear)

        # And a listener for the save button.
        self.control_panel.observe_save(self.on_save)

    # Tool Locking Methods -----------------------------------------------------

    def _lock_tool(self):   
        """Locks the annotation tool, preventing user interaction with the figure."""
        self.state.locked = True
        self.control_panel.figure_size_slider.disabled = True

        style_panel = self.control_panel.style_panel
        style_panel.style_dropdown.disabled = True
        for widget in style_panel.style_widgets.values():
            widget.disabled = True


    def _unlock_tool(self):
        """Unlocks the annotation tool, allowing user interaction with the figure."""
        self.state.locked = False
        self.control_panel.figure_size_slider.disabled = False

        style_panel = self.control_panel.style_panel
        style_panel.style_dropdown.disabled = False
        for widget in style_panel.style_widgets.values():
            widget.disabled = False
            
    # Figure Refresh Methods ---------------------------------------------------

    def refresh_figure(self):
        # Get the target and annotation.
        target_id     = self.control_panel.target
        annotation    = self.control_panel.annotation
        target_annots = self.state.annotations[target_id]

        # Check that the selected annotation has valid fixed annotations. 
        error = None
        fixed_points = self.annot_cfg.fixed_points[annotation]
        for fp in fixed_points: # for the name of the fixed point
            # Determine if the fixed point is a fixed head or tail. 
            fp_type = ( "fixed_head" if fp in 
                self.annot_cfg.fixed_head[annotation] else "fixed_tail" )
            
            # If there is no data for this fixed point or if the fixed point is
            # the only data for the annotation, then we have an error.
            if target_annots.is_lazy(fp) or target_annots[fp].shape[0] == 0:
                error = f"Annotation '{annotation}' requires fixed point '{fp}' " \
                        f"which is not yet available for target: {target_id}."
                break

            # If there is data for this fixed point, must make sure the fixed 
            # point is valid based on the annotation type. For contours and 
            # boundary, must have data besides their own fixed points (if there).
            atype = self.annot_cfg.types[fp] 
            if atype != "point": # atype in ( "contour", "boundary" )
                n_deps = len(self.annot_cfg.fixed_points[fp])
                if target_annots[fp].shape[0] <= n_deps:
                    error = f"Annotation '{annotation}' requires fixed point '{fp}' " \
                            f"which is not yet available for target: {target_id}."
                    break

            # If there is data for this fixed point, we need to make sure that 
            # the figure panel can calculate the fixed point based on the current data.
            try:
                self.figure_panel.calc_fixed_point(annotation, target_annots, fp_type)
            except Exception as e:
                error = f"Annotation '{annotation}' requires fixed point '{fp}' " \
                        f"which cannot be calculated for target: {target_id} " \
                        f"with the current data: {e}"
                break
    
        # If there was an error, we need to put an appropriate message up. 
        # Otherwise, we can clear any messages and just show the figure.
        if error is not None:
            # Lock the annotation tool, so user cannot interact with the figure.
            self._lock_tool()

            # Write the error message. 
            self.figure_panel.write_message(error)
        else:
            # Unlock the annotation tool, so user can interact with the figure.
            self._unlock_tool()
            
            # Clear any messages that might be up from before.
            self.figure_panel.clear_message()

            # Update the figure panel state variables.
            self.figure_panel.update_state(target_id, annotation, target_annots)

            # Redraw the figure. 
            self.figure_panel.redraw_canvas()

    # Event Handler Methods ----------------------------------------------------

    def on_selection_change(self, key, change):
        """This method runs when the control panel's selection changes."""
        if change.name != "value": return
        # First, things first: save the annotations.
        self.state.save_annotations()
        
        # Clear the save hooks if there are any.
        self.state.save_hooks = None

        # Update the control panel legend. 
        self.control_panel.legend_panel.update_legend(
            target_id  = self.control_panel.target,
            annotation = self.control_panel.annotation, 
        )

        # The selection has changed; we need to redraw the image and update the
        # annotations.
        self.refresh_figure()


    def on_figure_size_change(self, change):
        """This method runs when the control panel's figure size slider changes."""
        # Only respond to changes in the value of the style elements, 
        if change.name != "value": return

        # Update the state.
        self.state.figure_size(change.new)

        # Resize the figure panel. 
        self.figure_panel.resize_canvas(change.new)


    def on_style_change(self, annotation, key, change):
        """This method runs when the control panel's style elements change."""
        # Only respond to changes in the value of the style elements, 
        if change.name != "value": return
        
        # Update the state.
        self.state.style(annotation, { key: change.new })
        
        # Then redraw the annotation.
        self.figure_panel.redraw_canvas(redraw_image = False)


    def on_clear(self, button):
        """This method runs when the control panel's clear all button is clicked."""
        # The clear all button has a confirmation process. When the user first 
        # clicks it, it changes to a "Confirm Clear" button. If they click it
        # again, then the annotations are cleared. The button then resets to the
        # original "Clear All" state.
        if button.description == "Clear All":
            # Update the button to the confirmation state.
            button.description  = "Confirm Clear"
            button.button_style = "danger"
            
        elif button.description == "Confirm Clear":
            # Update the button back to the original state.
            button.description  = "Clear All"
            button.button_style = "warning"

            # Clear the annotations for the current target.
            target_id = self.control_panel.target
            for annotation in self.state.annotations[target_id].keys():
                self.state.annotations[target_id][annotation] = (
                    self.figure_panel.empty_point_matrix())

            # Refresh the figure to show the cleared annotations.
            self.refresh_figure()
        else:
            # If the button is in some unexpected state, we just reset it to the
            # original state.
            button.description  = "Clear All"
            button.button_style = "warning"

        
    def on_save(self, button):
        """This method runs when the control panel's save button is clicked."""
        self.state.save_annotations()
        self.state.save_preferences()