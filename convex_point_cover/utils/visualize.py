import numpy as np
from scipy.spatial import ConvexHull


def input_to_ascii(dataset):
    include_points = np.array(dataset["include"])
    exclude_points = np.array(dataset["exclude"])
    all_points = np.vstack((include_points, exclude_points))

    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)

    width = 50
    aspect_ratio = (max_y - min_y) / (max_x - min_x) / 2.0
    height = int(width * aspect_ratio)

    grid = [[" " for _ in range(width)] for _ in range(height)]

    def scale_point(point):
        x = int((point[0] - min_x) / (max_x - min_x) * (width - 1))
        y = int((point[1] - min_y) / (max_y - min_y) * (height - 1))
        return x, y

    for point in dataset["include"]:
        x, y = scale_point(point)
        grid[y][x] = "O"

    for point in dataset["exclude"]:
        x, y = scale_point(point)
        grid[y][x] = "X"

    return "\n".join("".join(row) for row in grid)


def output_to_ascii(dataset, output):
    include_points = np.array(dataset["include"])
    exclude_points = np.array(dataset["exclude"])
    all_points = np.vstack((include_points, exclude_points))

    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)

    width = 50
    aspect_ratio = (max_y - min_y) / (max_x - min_x) / 2.0
    height = int(width * aspect_ratio)

    grid = [[" " for _ in range(width)] for _ in range(height)]

    def scale_point(point):
        x = int((point[0] - min_x) / (max_x - min_x) * (width - 1))
        y = int((point[1] - min_y) / (max_y - min_y) * (height - 1))
        return x, y

    for partition in output:
        if len(partition) < 2:
            continue
        if len(partition) == 2:
            start = scale_point(partition[0])
            end = scale_point(partition[1])
            draw_line(grid, start, end, char=".")
            continue
        hull = ConvexHull(partition)
        for simplex in hull.simplices:
            start = scale_point(partition[simplex[0]])
            end = scale_point(partition[simplex[1]])
            draw_line(grid, start, end)

    for point in dataset["include"]:
        x, y = scale_point(point)
        grid[y][x] = "O"

    for point in dataset["exclude"]:
        x, y = scale_point(point)
        grid[y][x] = "X"

    return "\n".join("".join(row) for row in grid)


def draw_line(grid, start, end, char="#"):
    x1, y1 = start
    x2, y2 = end
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        grid[y1][x1] = char
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
