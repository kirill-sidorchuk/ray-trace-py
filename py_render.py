from typing import Optional

import numpy as np


class Camera:
    """
    Projects onto x-z plane. y-axis is the depth axis.
    """

    def __init__(
            self,
            position: np.ndarray,
            direction: np.ndarray,
            up: np.ndarray,
            perspective: float,
            width: int,
            height: int
        ):
        """
        :param position: [x, y, z] float
        :param direction:  [x, y, z] float (direction vector)
        :param up:  [x, y, z] float (up vector)
        :param perspective:  float
        """
        # normalize direction
        direction = direction / np.linalg.norm(direction)

        # normalize up
        up = up / np.linalg.norm(up)

        # calculate right vector
        right = np.cross(direction, up)

        # calculate new up vector (orthogonal to direction and right)
        up = np.cross(right, direction)

        self.position = position
        self.direction = direction
        self.up = up
        self.right = right
        self.perspective = perspective
        self.width = width
        self.height = height

        # calculate transformation matrix: from world space to camera space
        self.world_to_cam = np.array([right, direction, up], dtype=np.float32).T

    def project_point(self, point: np.ndarray) -> Optional[np.ndarray]:
        """
        Projects point onto x-z plane.

        :param point: [x, y, z] float
        :return: [x, y, depth] screen coordinates or None if point is behind camera
        """

        # translate point to camera space
        point = np.dot(self.world_to_cam, point - self.position)

        depth = point[1]
        if depth < 0:
            return None

        # project point onto x-z plane using central projection
        t = self.perspective / (depth + self.perspective)
        screen_x = point[0] * t + self.width / 2
        screen_y = self.height / 2 - point[2] * t

        return np.array([screen_x, screen_y, depth], dtype=np.float32)
