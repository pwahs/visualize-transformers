from scipy.spatial import ConvexHull, Delaunay


class PointSetWithConvexHull:
    # TODO: implement for more than 2 dimensions
    ## extends ConvexHull functionality to set of points less than the dimension+1
    def __init__(self, points):
        assert len(points) == 0 or len(points[0]) == 2
        self.points = points
        self.delaunay = None

    def add_point(self, point):
        self.points.append(point)
        self.delaunay = None

    def is_point_on_line_segment(self, a, b, c):
        # Check if point c is on the line segment between points a and b
        # Check is c is collinear with a and b
        cross_product = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(cross_product) > 1e-10:
            return False

        # Check is c is on the other side of a compared to b
        dot_product = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1]) * (b[1] - a[1])
        if dot_product < 0:
            return False

        # Check is c is further away from a than b
        squared_length_ab = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
        if dot_product > squared_length_ab:
            return False

        return True

    def is_inside_hull(self, point):
        if self.delaunay is None:
            if len(self.points) == 0:
                return False
            if len(self.points) == 1:
                return (self.points[0] == point).all()
            if len(self.points) == 2:
                return self.is_point_on_line_segment(
                    self.points[0], self.points[1], point
                )
            self.delaunay = Delaunay(ConvexHull(self.points).points)
        return self.delaunay.find_simplex(point) != -1
