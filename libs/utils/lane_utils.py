import numpy as np
from scipy.interpolate import splprep, splev, InterpolatedUnivariateSpline


class Lane:
    # Lane instance structure.
    # Adapted from:
    # https://github.com/lucastabelini/LaneATT/blob/main/lib/lane.py
    # Copyright (c) 2021 Lucas Tabelini
    def __init__(self, points=None, invalid_value=-2.0, metadata=None):
        super(Lane, self).__init__()
        self.curr_iter = 0
        self.points = points
        self.invalid_value = invalid_value
        self.function = InterpolatedUnivariateSpline(
            points[:, 1], points[:, 0], k=min(3, len(points) - 1)
        )
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01

        self.metadata = metadata or {}

    def __repr__(self):
        return '[Lane]\n' + str(self.points) + '\n[/Lane]'

    def __call__(self, lane_ys):
        lane_xs = self.function(lane_ys)

        lane_xs[(lane_ys < self.min_y) | (lane_ys > self.max_y)] = self.invalid_value
        return lane_xs

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_iter < len(self.points):
            self.curr_iter += 1
            return self.points[self.curr_iter - 1]
        self.curr_iter = 0
        raise StopIteration


def interp(points, n=5):
    """
    Adapted from:
    https://github.com/lucastabelini/LaneATT/blob/main/utils/culane_metric.py
    Copyright (c) 2021 Lucas Tabelini
    Args:
        points (List[tuple]): List of lane point tuples (x, y).
        n (int): number of interpolated points
    Returns:
        output (np.ndarray): Interpolated N lane points with shape (N, 2).
    """
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, u = splprep([x, y], s=0, t=5, k=min(3, len(points) - 1))

    u = np.linspace(0.0, 1.0, num=(len(u) - 1) * n + 1)
    output = np.array(splev(u, tck)).T
    return output
