from __future__ import annotations

from typing import Tuple, List
import os
import numpy as np
import json
from enum import Enum
from astropy.coordinates import cartesian_to_spherical
import math


class CoordinateSystem(Enum):

    CARTESIAN = 1
    SPHERICAL = 2
    BOTH = 3


class Point:
    """A point in Cartesian Coordinates in 3D"""

    def __init__(
        self, x: float = 0, y: float = 0, z: float = 0, spherical: bool = False
    ):
        """Constructor of 3D Point"""
        self.x = x
        self.y = y
        self.z = z
        self.spherical = spherical
        if spherical:
            r, phi_, theta_ = cartesian_to_spherical(self.x, self.y, self.z)
            self.r = r.value
            self.phi_ = phi_.value
            self.theta_ = theta_.value
        else:
            self.r = None
            self.phi_ = None
            self.theta_ = None

    def __add__(self, other: Point) -> Point:
        """Addition of 2 Points in Cartesian Coordinates"""
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Point) -> Point:
        """Subtraction of 2 Points in Cartesian Coordinates"""
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scale: float) -> Point:
        """Scales the Coordinates of Point"""
        return Point(self.x * scale, self.y * scale, self.z * scale)

    def length(self) -> float:
        """Distance of Point to the Origin"""
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

    def rotate(self, theta: float, phi: float) -> Point:
        """
        Rotates a Point Around the Origin For Given Theta and Phi
        Theta => Angle in Degrees : [0,360)
        Phi => Angle in Degrees : [-90,90]
        """
        if theta >= 360 or theta < 0:
            raise Exception("Theta must be between 0 and 360")

        if phi > 90 or phi < -90:
            raise Exception("Phi must be between -90 and 90")

        if not self.spherical:
            r, phi_, theta_ = cartesian_to_spherical(self.x, self.y, self.z)
            self.r = r.value
            self.phi_ = phi_.value
            self.theta_ = theta_.value
            self.spherical = True

        theta = theta * (np.pi / 180) + self.theta_
        phi = phi * (np.pi / 180) + self.phi_

        x = self.r * np.cos(phi) * np.cos(theta)
        y = self.r * np.cos(phi) * np.sin(theta)
        z = self.r * np.sin(phi)

        rotated_point = Point(x, y, z, False)

        rotated_point.r = self.r
        rotated_point.theta_ = theta
        rotated_point.phi_ = phi
        rotated_point.spherical = True

        return rotated_point

    def __contains__(self, z_coordinates: Tuple[int, List[float]]) -> bool:
        """
        Checks the projection of the point is in a given axis or not
        Selects the closest z-axis
        """
        return z_coordinates[0] == min(
            range(len(z_coordinates[1])),
            key=lambda i: abs(z_coordinates[1][i] - self.z),
        )

    def convert_numpy(self, coordinate_system: CoordinateSystem):
        """
        Representation of a Point in Numpy Array Format
        """
        if coordinate_system == CoordinateSystem.CARTESIAN:
            return np.array([self.x, self.y, self.z])
        elif coordinate_system == CoordinateSystem.SPHERICAL:

            if not self.spherical:
                r, phi_, theta_ = cartesian_to_spherical(self.x, self.y, self.z)
                self.r = r.value
                self.phi_ = phi_.value
                self.theta_ = theta_.value
                self.spherical = True

            return np.array([self.r, self.theta_, self.phi_])

        elif coordinate_system == CoordinateSystem.BOTH:
            return np.hstack(
                (
                    self.convert_numpy(CoordinateSystem.CARTESIAN),
                    self.convert_numpy(CoordinateSystem.SPHERICAL),
                )
            )


class Prism:
    def __init__(self, p1: Point = Point(), p2: Point = Point()):

        self.p1 = p1
        self.p2 = p2

    def __add__(self, other: Point) -> Prism:
        """
        Shifts the Center of the  Sphere
        """
        return Prism(self.p1 + other, self.p2 + other)

    def __sub__(self, other: Point) -> Prism:
        """
        Shifts the Center of the  Sphere
        """
        return Prism(self.center + other, self.radius)

    def __contains__(self, other: Point) -> bool:
        """
        Checks the Point is Inside the Sphere or Not
        """
        return True

    def rotate(self, theta: float, phi: float) -> Prism:
        """
        Rotates the Sphere Around the Origin
        Theta => Angle in Degrees : [0,360)
        Phi => Angle in Degrees : [-90,90]
        """
        return Prism(self.p1.rotate(theta, phi), self.p2.rotate(theta, phi))

    def scale(self, scale: float) -> Prism:
        """
        Scales the Center and Radius of the Sphere
        """
        return Prism(self.p1 * scale, self.p2 * scale)

    def convert_numpy(self, coordinate_system: CoordinateSystem):
        """
        Representation of a Point in Numpy Array Format
        """
        return np.hstack(
            (
                self.p1.convert_numpy(coordinate_system),
                self.p2.convert_numpy(coordinate_system),
            )
        )


class Sphere:
    """Sphere in 3D space."""

    def __init__(self, center: Point = Point(), radius: float = 1):
        """
        Constructor of Sphere
        Default is Unit Sphere on center
        """
        self.center = center
        self.radius = radius

    def __add__(self, other: Point) -> Sphere:
        """
        Shifts the Center of the  Sphere
        """
        return Sphere(self.center + other, self.radius)

    def __sub__(self, other: Point) -> Sphere:
        """
        Shifts the Center of the  Sphere
        """
        return Sphere(self.center + other, self.radius)

    def __contains__(self, other: Point) -> bool:
        """
        Checks the Point is Inside the Sphere or Not
        """
        return self.radius >= (self.center - other).length()

    def __contains__(self, other: Sphere) -> bool:
        """
        Checks two Spheres Intersects or not
        """
        return self.radius + other.radius >= (self.center - other.center).length()

    def rotate(self, theta: float, phi: float) -> Sphere:
        """
        Rotates the Sphere Around the Origin
        Theta => Angle in Degrees : [0,360)
        Phi => Angle in Degrees : [-90,90]
        """
        return Sphere(self.center.rotate(theta, phi), self.radius)

    def scale(self, scale: float) -> Sphere:
        """
        Scales the Center and Radius of the Sphere
        """
        return Sphere(self.center * scale, self.radius * scale)

    def projection(self, z_coordinate: float) -> Tuple[float, float, float]:
        """
        Finds the projection of Sphere
        Returns a tuple =>(center_x,center_y,radius)
        """
        radius = (self.radius**2 - (z_coordinate - self.center.z) ** 2) ** 0.5
        return (self.center.x, self.center.y, radius) if radius > 0 else None

    def convert_numpy(self, coordinate_system: CoordinateSystem):
        """
        Representation of a Point in Numpy Array Format
        """
        return np.hstack((self.center.convert_numpy(coordinate_system), self.radius))


class Topology:

    def __init__(
        self,
        intended_absorber: Sphere,
        absorbers: List[Sphere],
        reflectors: List[Sphere],
        absorber_prisms: List[Prism],
        reflector_prisms: List[Prism],
        diff_coef: float,
        line_of_sight: List[int],
    ):
        """Representation of 3D MCvD Channel"""
        self.intended_absorber = intended_absorber
        self.absorbers = absorbers
        self.reflectors = reflectors
        self.diff_coef = diff_coef
        self.absorber_prisms = absorber_prisms
        self.reflector_prisms = reflector_prisms
        # Calculate Time Step Value
        self.time_step = (
            (self.intended_absorber.center).length() - self.intended_absorber.radius
        ) ** 2 / (600 * diff_coef)
        self.line_of_sight = line_of_sight

    @staticmethod
    def parse_channel_config(channel_config: dict, intended_index: int):
        # List of absorbers (x coordinate, y coordinate, radius)
        absorbers = [
            Sphere(Point(absorber[0], absorber[1], absorber[2], True), absorber[3])
            for absorber in channel_config["absorbers"]
        ]

        # List of reflectors (x coordinate, y coordinate, radius)
        reflectors = [
            Sphere(Point(reflector[0], reflector[1], reflector[2], True), reflector[3])
            for reflector in channel_config["reflectors"]
        ]

        absorber_prisms = [
            Prism(
                Point(absorber_prism[0], absorber_prism[1], absorber_prism[2], True),
                Point(absorber_prism[3], absorber_prism[4], absorber_prism[5], True),
            )
            for absorber_prism in channel_config["absorber_prisms"]
        ]

        reflector_prisms = [
            Prism(
                Point(reflector_prism[0], reflector_prism[1], reflector_prism[2], True),
                Point(reflector_prism[3], reflector_prism[4], reflector_prism[5], True),
            )
            for reflector_prism in channel_config["reflector_prisms"]
        ]

        # Diffusion Coefficient
        diff_coef = channel_config["diff_coef"]

        # line_of_sight = channel_config["line_of_sight"][intended_index]

        return Topology(
            absorbers[intended_index],
            absorbers[:intended_index] + absorbers[intended_index + 1 :],
            reflectors,
            absorber_prisms,
            reflector_prisms,
            diff_coef,
            None,
        )

    @staticmethod
    def read_config(input_txt_path: str, intended_index: int) -> Topology:
        """Read Configuration from input file"""
        if not os.path.isfile(input_txt_path):
            raise Exception("Configuration file cannot be found.")

        # Read configuration of the channel
        topology_dict = json.load(open(input_txt_path, "r"))

        return Topology.parse_channel_config(topology_dict, intended_index)

    def convert_numpy(
        self,
        coordinate_system: CoordinateSystem,
        max_spherical_entity: int,
        order: str,
        flatten: bool = True,
        one_absorber_points: int = 0,
        line_of_sight: bool = False,
    ):
        """
        Representation of a Channel Topology in Numpy Array Format
        """
        numpy_repr = np.hstack(
            (0, self.intended_absorber.convert_numpy(coordinate_system), 0, 0, 0, 0, 0)
        )

        entities = np.zeros((0, numpy_repr.shape[0]))
        # Insert Other Absorbers
        for a in self.absorbers:
            entities = np.vstack(
                (
                    entities,
                    np.hstack((1, a.convert_numpy(coordinate_system), 0, 0, 0, 0, 0)),
                )
            )

        # Insert Reflectors
        for r in self.reflectors:
            entities = np.vstack(
                (
                    entities,
                    np.hstack((2, r.convert_numpy(coordinate_system), 0, 0, 0, 0, 0)),
                )
            )

        for a_p in self.absorber_prisms:
            entities = np.vstack(
                (entities, np.hstack((3, a_p.convert_numpy(coordinate_system))))
            )
        for r_p in self.reflector_prisms:
            entities = np.vstack(
                (entities, np.hstack((4, r_p.convert_numpy(coordinate_system))))
            )

        num_empty_entities = (
            max_spherical_entity
            - len(self.reflectors)
            - len(self.absorbers)
            - 1
            - len(self.absorber_prisms)
            - len(self.reflector_prisms)
        )
        if num_empty_entities != 0:
            # 0's for position -1 for Class (Empty Entities)
            entities = np.vstack(
                (
                    entities,
                    np.array(
                        [
                            [-1] + [0 for i in range(numpy_repr.shape[0] - 1)]
                            for i in range(num_empty_entities)
                        ]
                    ),
                )
            )

        if order == "shuffle":
            np.random.shuffle(entities)

        numerical_inputs = [self.diff_coef, self.time_step] + self.one_absorber_f_hit(
            one_absorber_points
        )

        if line_of_sight:
            numerical_inputs += [
                self.line_of_sight[i] / self.line_of_sight[-1]
                for i in range(len(self.line_of_sight) - 1)
            ]

        if flatten:
            return np.hstack((numpy_repr, entities.flatten(), numerical_inputs))
        else:
            # Insert Diffusion Coefficient and Time Step
            return np.vstack((numpy_repr, entities)), np.array(numerical_inputs)

    def one_absorber_f_hit(self, one_absorber_points: int):
        f_hit = []

        for i in range(one_absorber_points):
            t = (i + 1) * (800 / one_absorber_points) * self.time_step
            r_r = self.intended_absorber.radius
            r_0 = self.intended_absorber.center.length()
            f_hit.append(
                (r_r / r_0) * math.erfc((r_0 - r_r) / math.sqrt(4 * self.diff_coef * t))
            )

        return f_hit

    def set_time_output(self, time_output):
        """
        Store CIR Integral inside the Topology Class
        """
        self.time_output = time_output

    def rotate(self, theta: float, phi: float) -> Topology:
        """
        Rotate the Topology Around the Point Transmitter
        """

        absorbers = [absorber.rotate(theta, phi) for absorber in self.absorbers]
        reflectors = [reflector.rotate(theta, phi) for reflector in self.reflectors]
        absorber_prisms = [
            absorber_prism.rotate(theta, phi) for absorber_prism in self.absorber_prisms
        ]
        reflector_prisms = [
            reflector_prism.rotate(theta, phi)
            for reflector_prism in self.reflector_prisms
        ]

        return Topology(
            self.intended_absorber.rotate(theta, phi),
            absorbers,
            reflectors,
            absorber_prisms,
            reflector_prisms,
            self.diff_coef,
            self.line_of_sight,
        )

    def scale(self, scale: float) -> Topology:
        """
        Scales the Topology Coordinates
        """

        absorbers = [absorber.scale(scale) for absorber in self.absorbers]
        reflectors = [reflector.scale(scale) for reflector in self.reflectors]

        return Topology(
            self.intended_absorber.scale(scale), absorbers, reflectors, self.diff_coef
        )

    def visualize(self):
        from .visualizer import visualize_topology

        visualize_topology(self)

    def projection(self, z_coordinates: Tuple[int, List[float]]) -> Tuple[
        Tuple[float, float],
        Tuple[float, float, float],
        List[Tuple[float, float, float]],
        List[Tuple[float, float, float]],
    ]:
        """
        Calculates Projection of the Topology
        """
        # Control Point Transmitter
        transmitter = Point(0, 0, 0)
        transmitter_projection = None
        if transmitter in z_coordinates:
            transmitter_projection = (transmitter.x, transmitter.y)

        z_coordinate = z_coordinates[1][z_coordinates[0]]

        intended_absorber_projection = self.intended_absorber.projection(z_coordinate)
        absorber_projections = [
            absorber.projection(z_coordinate) for absorber in self.absorbers
        ]
        reflector_projections = [
            reflector.projection(z_coordinate) for reflector in self.reflectors
        ]

        return (
            transmitter_projection,
            intended_absorber_projection,
            absorber_projections,
            reflector_projections,
        )
