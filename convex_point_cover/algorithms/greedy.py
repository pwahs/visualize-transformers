import numpy as np

from utils.PointSetWithConvexHull import PointSetWithConvexHull


def greedy(points, exclude):
    points = np.array(points)
    exclude = np.array(exclude)
    convex_sets = []
    current_set = np.array([])
    for p in points:
        current_set = (
            np.append(current_set, [p], axis=0) if current_set.size else np.array([p])
        )
        point_set = PointSetWithConvexHull(current_set)
        for e in exclude:
            if point_set.is_inside_hull(e):
                current_set = current_set[:-1]
                convex_sets.append(current_set)
                current_set = np.array([p])
                break
    convex_sets.append(current_set)
    return convex_sets
