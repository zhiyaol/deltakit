import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly.graph_objects as go
import itertools
import numpy as np
from pathlib import Path
from typing import Optional

import itertools
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Optional, Set, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
from deltakit_circuit import Qubit
from deltakit_circuit.basic_types import Coord2D, Coord2DDelta
from deltakit_circuit.qubit_identifiers import PauliGate, PauliX, PauliZ
from deltakit_explorer.codes.css._css_code import CSSCode
from deltakit_explorer.codes.stabiliser import Stabiliser
from deltakit_explorer.enums.basic_enums import DrawingColours

def draw_patch(self, dataQubitClickCallback = lambda _: None, ancillaQubitClickCallback = lambda _: None, filename: Optional[str] = None, unrotated_code: bool = False) -> go.FigureWidget:
    """
    Draw an interactive picture of the planar code using Plotly FigureWidget.
    Clicking on qubits will highlight them.

    Parameters
    ----------
    filename: str, optional
        Path to the file where to save the pictorial representation of the
        planar code stored in this class. Will save as HTML format.
        
    Returns
    -------
    plotly.graph_objects.FigureWidget
        The interactive Plotly FigureWidget object containing the visualization.
    """
    all_qubit_x_coords = [qubit.unique_identifier.x for qubit in self.qubits]
    all_qubit_y_coords = [qubit.unique_identifier.y for qubit in self.qubits]
    diff_from_max_coord_to_margin_no_ancilla = (
        2 if not unrotated_code or not (self.linear_tr == np.eye(2)).all() else 1
    )
    
    if self._use_ancilla_qubits:
        min_x, max_x = min(all_qubit_x_coords) - 1, max(all_qubit_x_coords) + 1
        min_y, max_y = min(all_qubit_y_coords) - 1, max(all_qubit_y_coords) + 1
    else:
        min_x, max_x = (
            min(all_qubit_x_coords) - diff_from_max_coord_to_margin_no_ancilla,
            max(all_qubit_x_coords) + diff_from_max_coord_to_margin_no_ancilla,
        )
        min_y, max_y = (
            min(all_qubit_y_coords) - diff_from_max_coord_to_margin_no_ancilla,
            max(all_qubit_y_coords) + diff_from_max_coord_to_margin_no_ancilla,
        )
    
    x_lim = (min_x, max_x)
    y_lim = (min_y, max_y)
    
    # Prepare trace data
    traces = []
    
    stabilisers = tuple(itertools.chain.from_iterable(self._stabilisers))
    stabilisers = self._sort_stabilisers(stabilisers)
    
    # Draw stabiliser plaquettes
    for stabiliser in stabilisers:
        data_qubit_x_coords = [
            pauli.qubit.unique_identifier[0]
            for pauli in stabiliser.paulis
            if pauli is not None
        ]
        data_qubit_y_coords = [
            pauli.qubit.unique_identifier[1]
            for pauli in stabiliser.paulis
            if pauli is not None
        ]
        
        paulis = [pauli for pauli in stabiliser.paulis if pauli is not None]
        
        if len(paulis) == 2:
            ancilla_coord = stabiliser.ancilla_qubit.unique_identifier
            data_qubit_x_coords.append(ancilla_coord[0])
            data_qubit_y_coords.append(ancilla_coord[1])
        elif len(paulis) == 4:
            data_qubit_x_coords[2], data_qubit_x_coords[3] = (
                data_qubit_x_coords[3],
                data_qubit_x_coords[2],
            )
            data_qubit_y_coords[2], data_qubit_y_coords[3] = (
                data_qubit_y_coords[3],
                data_qubit_y_coords[2],
            )
        
        # Close the polygon by adding the first point at the end
        data_qubit_x_coords.append(data_qubit_x_coords[0])
        data_qubit_y_coords.append(data_qubit_y_coords[0])
        
        if isinstance(paulis[0], PauliX):
            fill_color = DrawingColours.X_COLOUR.value
            name = 'X Stabilizer'
        else:
            fill_color = DrawingColours.Z_COLOUR.value
            name = 'Z Stabilizer'
        
        # Add filled polygon for stabiliser plaquette
        traces.append(go.Scatter(
            x=data_qubit_x_coords,
            y=data_qubit_y_coords,
            fill='toself',
            fillcolor=fill_color,
            line=dict(color=fill_color, width=1),
            mode='lines',
            name=name,
            showlegend=False,
            hovertemplate=f"{name}<extra></extra>"
        ))
    
    # Prepare data qubits coordinates
    data_qubit_x = [qubit.unique_identifier[0] for qubit in self._data_qubits]
    data_qubit_y = [qubit.unique_identifier[1] for qubit in self._data_qubits]
    
    # Add data qubits trace
    traces.append(go.Scatter(
        x=data_qubit_x,
        y=data_qubit_y,
        mode='markers',
        marker=dict(
            size=20,
            color=DrawingColours.DATA_QUBIT_COLOUR.value,
            line=dict(color=DrawingColours.DATA_QUBIT_COLOUR.value, width=1)
        ),
        name='Data Qubits',
        showlegend=True,
        hovertemplate='Data Qubit (%{x}, %{y})<br>Click to highlight<extra></extra>'
    ))
    
    # Store trace indices for later reference
    data_qubit_trace_idx = len(traces) - 1
    x_ancilla_trace_idx = None
    z_ancilla_trace_idx = None
    
    if self._use_ancilla_qubits:
        # Draw X stabiliser ancilla qubits
        if self._x_ancilla_qubits:
            x_ancilla_x = [qubit.unique_identifier[0] for qubit in self._x_ancilla_qubits]
            x_ancilla_y = [qubit.unique_identifier[1] for qubit in self._x_ancilla_qubits]
            
            traces.append(go.Scatter(
                x=x_ancilla_x,
                y=x_ancilla_y,
                mode='markers',
                marker=dict(
                    size=20,
                    color=DrawingColours.ANCILLA_QUBIT_COLOUR.value,
                    line=dict(color=DrawingColours.ANCILLA_QUBIT_COLOUR.value, width=1)
                ),
                name='X Ancilla Qubits',
                showlegend=True,
                hovertemplate='X Ancilla Qubit (%{x}, %{y})<br>Click to highlight<extra></extra>'
            ))
            x_ancilla_trace_idx = len(traces) - 1
        
        # Draw Z stabiliser ancilla qubits
        if self._z_ancilla_qubits:
            z_ancilla_x = [qubit.unique_identifier[0] for qubit in self._z_ancilla_qubits]
            z_ancilla_y = [qubit.unique_identifier[1] for qubit in self._z_ancilla_qubits]
            
            traces.append(go.Scatter(
                x=z_ancilla_x,
                y=z_ancilla_y,
                mode='markers',
                marker=dict(
                    size=20,
                    color=DrawingColours.ANCILLA_QUBIT_COLOUR.value,
                    line=dict(color=DrawingColours.ANCILLA_QUBIT_COLOUR.value, width=1)
                ),
                name='Z Ancilla Qubits',
                showlegend=True,
                hovertemplate='Z Ancilla Qubit (%{x}, %{y})<br>Click to highlight<extra></extra>'
            ))
            z_ancilla_trace_idx = len(traces) - 1
    
    # Create FigureWidget with all traces
    f = go.FigureWidget(traces)
    
    # Set up interactivity for data qubits
    data_scatter = f.data[data_qubit_trace_idx]
    data_colors = [DrawingColours.DATA_QUBIT_COLOUR.value] * len(data_qubit_x)
    data_sizes = [20] * len(data_qubit_x)
    data_scatter.marker.color = data_colors
    data_scatter.marker.size = data_sizes
    
    pauli_colors = ['red', 'green', 'blue']
    
    # Create callback function for data qubits
    def update_data_point(trace, points, selector):
        c = list(data_scatter.marker.color)
        s = list(data_scatter.marker.size)
        for i in points.point_inds:
            if c[i] == pauli_colors[-1]: # If on last Pauli color, unhighlight point
                c[i] = DrawingColours.DATA_QUBIT_COLOUR.value
                s[i] = 20
            else:  
                if c[i] in pauli_colors: # If a Pauli color, change to next
                    c[i] = pauli_colors[pauli_colors.index(c[i]) + 1]
                else: # If not a Pauli color, change to first Pauli color
                    c[i] = pauli_colors[0]
                s[i] = 30
        with f.batch_update():
            data_scatter.marker.color = c
            data_scatter.marker.size = s
        dataQubitClickCallback(points)
    
    data_scatter.on_click(update_data_point)
    
    # Set up interactivity for X ancilla qubits
    if x_ancilla_trace_idx is not None:
        x_ancilla_scatter = f.data[x_ancilla_trace_idx]
        x_ancilla_colors = [DrawingColours.ANCILLA_QUBIT_COLOUR.value] * len(x_ancilla_x)
        x_ancilla_sizes = [20] * len(x_ancilla_x)
        x_ancilla_scatter.marker.color = x_ancilla_colors
        x_ancilla_scatter.marker.size = x_ancilla_sizes
        
        def update_x_ancilla_point(trace, points, selector):
            c = list(x_ancilla_scatter.marker.color)
            s = list(x_ancilla_scatter.marker.size)
            for i in points.point_inds:
                if c[i] == pauli_colors[-1]: # If on last Pauli color, unhighlight point
                    c[i] = DrawingColours.ANCILLA_QUBIT_COLOUR.value
                    s[i] = 20
                else:
                    if c[i] in pauli_colors: # If a Pauli color, change to next
                        c[i] = pauli_colors[pauli_colors.index(c[i]) + 1]
                    else: # If not a Pauli color, change to first Pauli color
                        c[i] = pauli_colors[0]
                    s[i] = 30
            with f.batch_update():
                x_ancilla_scatter.marker.color = c
                x_ancilla_scatter.marker.size = s
            ancillaQubitClickCallback(points)
        
        x_ancilla_scatter.on_click(update_x_ancilla_point)
    
    # Set up interactivity for Z ancilla qubits
    if z_ancilla_trace_idx is not None:
        z_ancilla_scatter = f.data[z_ancilla_trace_idx]
        z_ancilla_colors = [DrawingColours.ANCILLA_QUBIT_COLOUR.value] * len(z_ancilla_x)
        z_ancilla_sizes = [20] * len(z_ancilla_x)
        z_ancilla_scatter.marker.color = z_ancilla_colors
        z_ancilla_scatter.marker.size = z_ancilla_sizes
        
        def update_z_ancilla_point(trace, points, selector):
            c = list(z_ancilla_scatter.marker.color)
            s = list(z_ancilla_scatter.marker.size)
            for i in points.point_inds:
                if c[i] == pauli_colors[-1]: # If on last Pauli color, unhighlight point
                    c[i] = DrawingColours.ANCILLA_QUBIT_COLOUR.value
                    s[i] = 20
                else:
                    if c[i] in pauli_colors: # If a Pauli color, change to next
                        c[i] = pauli_colors[pauli_colors.index(c[i]) + 1]
                    else: # If not a Pauli color, change to first Pauli color
                        c[i] = pauli_colors[0]
                    s[i] = 30
            with f.batch_update():
                z_ancilla_scatter.marker.color = c
                z_ancilla_scatter.marker.size = s
            ancillaQubitClickCallback(points)
        
        z_ancilla_scatter.on_click(update_z_ancilla_point)
    
    # Update layout
    f.layout.update(
        title={
            'text': 'Interactive Planar Code Visualization<br><sub>Click on qubits to highlight them</sub>',
            # 'x': 0.5,  # Center the title
            # 'xanchor': 'center',
            # 'yanchor': 'top',
            'font': dict(size=20)
        },
        xaxis=dict(
            range=x_lim,
            showgrid=True,
            gridcolor='lightgray',
            scaleanchor='y',
            scaleratio=1,
            title='X Coordinate'
        ),
        yaxis=dict(
            range=y_lim,
            showgrid=True,
            gridcolor='lightgray',
            title='Y Coordinate'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        width=600,
        height=600,
        hovermode='closest'
    )
    
    # Save the file if filename is provided
    if filename:
        output_directory = Path(filename)
        if not output_directory.exists():
            output_directory.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as HTML (Plotly's native format)
        if not filename.endswith('.html'):
            filename = filename + '.html'
        f.write_html(filename)
        
        # Optionally save as static image (requires kaleido)
        # Uncomment the lines below if you want static images
        # static_filename = filename.replace('.html', '.png')
        # f.write_image(static_filename)
    
    return f

# Example of how to use the function:
  
# from deltakit.explorer.codes import css_code_memory_circuit, RotatedPlanarCode
# from deltakit.circuit.gates import PauliBasis
# import matplotlib.pyplot as plt

# rp_code = RotatedPlanarCode(width=5, height=5)
# deltakit_circuit = css_code_memory_circuit(
#     css_code=rp_code,
#     num_rounds=5,
#     logical_basis=PauliBasis.Z,
# )
# draw_patch(rp_code)
