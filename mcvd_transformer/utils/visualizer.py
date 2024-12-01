import plotly.graph_objects as go
import numpy as np

from .objects import Point, Sphere, Prism, Topology

from enum import Enum


class Color(Enum):

    INTENDED_ABSORBER = "red"
    REFLECTOR = "green"
    ABSORBER = "black"


def create_spherical_trace(spherical_entity: Sphere, color: str):
    """
    Creates Plotly Trace For Spherical Objects
    """
    # Set up 100 points. First, do angles
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)

    # Set up coordinates for points on the sphere
    x0 = spherical_entity.center.x + spherical_entity.radius * np.outer(
        np.cos(theta), np.sin(phi)
    )
    y0 = spherical_entity.center.y + spherical_entity.radius * np.outer(
        np.sin(theta), np.sin(phi)
    )
    z0 = spherical_entity.center.z + spherical_entity.radius * np.outer(
        np.ones(100), np.cos(phi)
    )
    # Set up trace
    trace = go.Surface(
        x=x0, y=y0, z=z0, colorscale=[[0, color.value], [1, color.value]]
    )
    trace.update(showscale=False)

    return trace


def create_prism_trace(prism: Prism, color: str):
    x0 = [
        prism.p1.x,
        prism.p1.x,
        prism.p1.x,
        prism.p1.x,
        prism.p2.x,
        prism.p2.x,
        prism.p2.x,
        prism.p2.x,
    ]
    y0 = [
        prism.p1.y,
        prism.p1.y,
        prism.p2.y,
        prism.p2.y,
        prism.p1.y,
        prism.p1.y,
        prism.p2.y,
        prism.p2.y,
    ]
    z0 = [
        prism.p1.z,
        prism.p2.z,
        prism.p1.z,
        prism.p2.z,
        prism.p1.z,
        prism.p2.z,
        prism.p1.z,
        prism.p2.z,
    ]

    trace = go.Mesh3d(
        x=x0,
        y=y0,
        z=z0,
        intensity=np.linspace(0, 1, 8, endpoint=True),
        alphahull=4,
        colorscale=[[0, color.value], [1, color.value]],
    )

    trace.update(showscale=False)

    return trace


def create_point_trace(point: Point, color: str):
    """
    Creates Plotly Trace For Point Objects
    """
    # Set up 100 points. First, do angles
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)

    # Set up coordinates for points on the sphere
    x0 = point.x + 0.2 * np.outer(np.cos(theta), np.sin(phi))
    y0 = point.y + 0.2 * np.outer(np.sin(theta), np.sin(phi))
    z0 = point.z + 0.2 * np.outer(np.ones(100), np.cos(phi))

    # Set up trace
    trace = go.Surface(
        x=x0, y=y0, z=z0, colorscale=[[0, color.value], [1, color.value]]
    )
    trace.update(showscale=False)

    return trace


def visualize_topology(topology: Topology):
    """
    Creates 3D Visualization of the MCvD Topology
    Assumes first absorber is the intended absorber
    """
    transmitter_trace = create_point_trace(Point(0, 0, 0), Color.INTENDED_ABSORBER)
    intended_absorber_trace = create_spherical_trace(
        topology.intended_absorber, Color.INTENDED_ABSORBER
    )
    other_absorber_traces = [
        create_spherical_trace(absorber, Color.ABSORBER)
        for absorber in topology.absorbers
    ]
    reflector_traces = [
        create_spherical_trace(reflector, Color.REFLECTOR)
        for reflector in topology.reflectors
    ]
    absorber_prism_trace = [
        create_prism_trace(absorber_prism, Color.ABSORBER)
        for absorber_prism in topology.absorber_prisms
    ]
    reflector_prism_trace = [
        create_prism_trace(reflector_prism, Color.REFLECTOR)
        for reflector_prism in topology.reflector_prisms
    ]

    layout = go.Layout(
        title="",
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(title="X", visible=False),
            yaxis=dict(title="Y", visible=False),
            zaxis=dict(title="Z", visible=False),
        ),
    )

    fig = go.Figure(
        data=[transmitter_trace, intended_absorber_trace]
        + other_absorber_traces
        + reflector_traces
        + absorber_prism_trace
        + reflector_prism_trace,
        layout=layout,
    )

    fig.show()
