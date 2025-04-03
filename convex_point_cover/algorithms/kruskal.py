import numpy as np

from scipy.spatial import ConvexHull
from itertools import combinations

from convex_point_cover.utils.PointSetWithConvexHull import PointSetWithConvexHull

from convex_point_cover.utils.visualize import output_to_ascii


# Inspired by Kruskal's algorithm: Try to add edges in increasing order of distance,
# but only if the new larger component does not contain an excluded point in it's convex hull
def kruskal(points, exclude):
    # Find root point for each connected component
    def find(parent, i):
        if parent[i] == i:
            return i
        return find(parent, parent[i])

    # Merge two connected components
    def union(parent, rank, x, y):
        root_x = find(parent, x)
        root_y = find(parent, y)
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

    # Convert parent-trees to partitioning of points into connected components
    def group_points_by_component(parent, points):
        components = {}
        for i in range(len(points)):
            root = find(parent, i)
            if root not in components:
                components[root] = []
            components[root].append(points[i])

        return list(components.values())

    # Generate all edges, sort by length
    edges = []
    for i, j in combinations(range(len(points)), 2):
        dist = np.linalg.norm(points[i] - points[j])
        edges.append((dist, i, j))

    edges.sort()
    parent = np.array(list(range(len(points))))
    rank = [0] * len(points)

    for edge in edges:
        dist, u, v = edge
        set_u = find(parent, u)
        set_v = find(parent, v)
        if set_u != set_v:
            new_points = np.array(
                [
                    points[i]
                    for i in range(len(points))
                    if find(parent, i) in [set_u, set_v]
                ]
            )

            hull = PointSetWithConvexHull(new_points)
            if not any(hull.is_inside_hull(point) for point in exclude):
                union(parent, rank, set_u, set_v)
                # print()
                # print(
                #     output_to_ascii(
                #         {"include": points, "exclude": exclude},
                #         group_points_by_component(parent, points),
                #     )
                # )

    return group_points_by_component(parent, points)
