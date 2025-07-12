from .plotter import GenericPlotter
import plotly.io as pio
import os
import json

def create_plot(data, plot_type, **kwargs):
    plotter = GenericPlotter()
    return plotter.create_plot(data, plot_type, **kwargs)
