import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from convex_point_cover.utils.visualize import input_to_ascii


def generate_random_dataset(n_in, n_out, opt, seed=None):
    np.random.seed(seed)
    # Generate opt random convex sets in [0,1]^2
    convex_hulls = []
    while len(convex_hulls) < opt:
        n = np.random.randint(5, 10)
        points = np.random.rand(n, 2)
        hull = ConvexHull(points)
        new_hull = Delaunay(hull.points)

        # Check for intersection with existing hulls. For simplicity, we only check if the vertices
        # are covered. Not perfect, but good enough.
        intersects = any(
            any(new_hull.find_simplex(point) >= 0 for point in existing_hull.points)
            for existing_hull in convex_hulls
        )

        if not intersects:
            convex_hulls.append(new_hull)

    # Generate random points, check if they are in the convex sets
    include = []
    exclude = []
    c_in = 0
    c_out = 0
    while c_in < n_in or c_out < n_out:
        point = np.random.rand(2)
        in_any_hull = any(hull.find_simplex(point) >= 0 for hull in convex_hulls)
        if in_any_hull and c_in < n_in:
            c_in += 1
            include.append(point)
        elif not in_any_hull and c_out < n_out:
            c_out += 1
            exclude.append(point)

    return {"include": np.array(include), "exclude": np.array(exclude), "optimal": opt}


def summarize(dataset):
    num_include = len(dataset["include"])
    num_exclude = len(dataset["exclude"])
    optimal = dataset["optimal"]
    return f"In:{num_include},Out:{num_exclude},Opt:{optimal}"


def get_datasets():
    dataset1 = {
        "include": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "exclude": np.array([[0.4, 0.5], [0.6, 0.5]]),
        "optimal": 2,
    }
    dataset2 = {
        "include": np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]),
        "exclude": np.array(
            [[0.5, 0], [0, 0.5], [1.5, 0], [1, 0.5], [2, 0.5], [0.5, 1], [1.5, 1]]
        ),
        "optimal": 4,
    }
    return [
        dataset1,
        dataset2,
        generate_random_dataset(20, 20, 4, seed=0),
        generate_random_dataset(100, 100, 10, seed=0),
        generate_random_dataset(20, 100, 6, seed=0),
    ]


if __name__ == "__main__":
    datasets = get_datasets()
    for i, dataset in enumerate(datasets):
        print(f"Dataset {i+1}:")
        print("Include points:")
        print(dataset["include"])
        print("Exclude points:")
        print(dataset["exclude"])
        print("Optimal number of convex sets:")
        print(dataset["optimal"])
        print()
        print(input_to_ascii(dataset))
        print()
