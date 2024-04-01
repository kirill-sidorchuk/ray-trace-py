import numpy as np
import cv2

from py_render import Camera


def create_ball(x: float, y: float, z: float, r: float, n: int, rotation: float) -> np.ndarray:
    """
    Create a point cloud of a ball. Points are distributed uniformly on the surface of the ball.

    :param x: center x coordinate
    :param y: center y coordinate
    :param z: center z coordinate
    :param r: radius
    :param n: number of points on the surface
    :param rotation: rotation angle in radians
    :return: [n, 3] float. point cloud
    """

    # reset random seed
    np.random.seed(42)

    # generate random points on the surface of a unit sphere
    phi = np.random.uniform(0, 2 * np.pi, n) + rotation
    cos_theta = np.random.uniform(-1, 1, n)
    theta = np.arccos(cos_theta)

    # convert spherical coordinates to cartesian
    x = r * np.sin(theta) * np.cos(phi) + x
    y = r * np.sin(theta) * np.sin(phi) + y
    z = r * np.cos(theta) + z

    return np.stack([x, y, z], axis=1)


def main():
    camera = Camera(
        position=np.array([0, 0, 0], dtype=np.float32),
        direction=np.array([0, 1, 0], dtype=np.float32),
        up=np.array([0, 0, 1], dtype=np.float32),
        perspective=400.0,
        width=800,
        height=600
    )

    canvas = np.zeros((camera.height, camera.width, 3), dtype=np.uint16)
    old_canvas = canvas.copy()
    rotation = 0.0

    while True:
        ball = create_ball(0, 500, 0, 300, 2000, rotation)

        canvas.fill(0)
        for point in ball:
            screen_coords = camera.project_point(point)
            if screen_coords is not None:
                x, y, depth = screen_coords
                color = int(min(255, 25500 / (depth * 0.25 + 1)))
                canvas[int(y), int(x)] = [color, color, color]

        # show canvas with opencv
        old_canvas = (canvas + old_canvas) // 2
        cv2.imshow('Point cloud', old_canvas.astype(np.uint8))
        k = cv2.waitKey(1)
        if k == 27:
            break

        rotation += 0.01
        if rotation > 2 * np.pi:
            rotation -= 2 * np.pi


if __name__ == '__main__':
    main()
