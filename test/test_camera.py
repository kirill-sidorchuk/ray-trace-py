import unittest

import numpy as np

from py_render import Camera


class TestCamera(unittest.TestCase):
    def test_camera(self):
        direction = np.array([0, 1, 0], dtype=np.float32)
        up = np.array([0, 0, 1], dtype=np.float32)
        position = np.array([0, 0, 0], dtype=np.float32)
        perspective = 1.0
        width = 640
        height = 480
        camera = Camera(position, direction, up, perspective, width, height)
        self.assertAlmostEqual(np.linalg.norm(camera.direction - direction), 0)
        self.assertAlmostEqual(np.linalg.norm(camera.up - up), 0)

        # test projection
        point = np.array([0, 1, 0], dtype=np.float32)
        projected = camera.project_point(point)
        self.assertIsNotNone(projected)
        self.assertEqual(projected[0], 320)
        self.assertEqual(projected[1], 240)
