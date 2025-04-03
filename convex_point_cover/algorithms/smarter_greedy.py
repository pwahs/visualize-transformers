import numpy as np

from utils.PointSetWithConvexHull import PointSetWithConvexHull


def smarter_greedy(points, exclude):
    exclude = np.array(exclude)
    convex_sets = []
    current_set = np.array([])
    points_left = np.array(points)
    while points_left.size:
        points = points_left
        points_left = np.array([])
        for p in points:
            current_set = (
                np.append(current_set, [p], axis=0)
                if current_set.size
                else np.array([p])
            )
            point_set = PointSetWithConvexHull(current_set)
            for e in exclude:
                if point_set.is_inside_hull(e):
                    current_set = current_set[:-1]
                    points_left = (
                        np.append(points_left, [p], axis=0)
                        if points_left.size
                        else np.array([p])
                    )
                    break
        convex_sets.append(current_set)
        current_set = np.array([])
    return convex_sets
