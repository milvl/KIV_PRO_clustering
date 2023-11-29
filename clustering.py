import random
import math
import timeit
from copy import deepcopy
from typing import List, Tuple, Dict, Callable
from xypoint import XYPoint
from xypoint import euclidean_distance, manhattan_distance, cosine_similarity

random.seed(random.randint(0, 1000000))

def create_dataset(n: int, k: int, delta: float, min_xy: XYPoint, max_xy: XYPoint) -> Dict[XYPoint, List[XYPoint]]:
    """
    Creates a dataset of n points with k clusters with deviation delta.

    :param n: The number of points
    :type n: int
    :param k: The number of clusters
    :type k: int
    :param delta: The deviation
    :type delta: float
    :param min_xy: The minimum x and y values
    :type min_xy: XYPoint
    :param max_xy: The maximum x and y values
    :type max_xy: XYPoint
    :return: The dataset
    :rtype: List[XYPoint]
    """

    # sanity checks
    if n < k:
        raise ValueError("n must be greater than k")
    if not delta:
        delta = int(random.uniform(0.1, 0.7)*math.sqrt((max_xy.x-min_xy.x) * (max_xy.y-min_xy.y)/math.pi*k))
    if delta <= 0:
        raise ValueError("delta must be greater than 0")
    if min_xy.x >= max_xy.x or min_xy.y >= max_xy.y:
        raise ValueError("min_xy must be less than max_xy")
    if delta >= euclidean_distance(min_xy, max_xy):
        raise ValueError("delta must be less than the distance between min_xy and max_xy")

    count = 0

    # generating centers
    centers = []
    for _ in range(k):
        suitable = False
        i = -1
        while not suitable:
            new_point = XYPoint(random.uniform(min_xy.x + delta, max_xy.x - delta), random.uniform(min_xy.y + delta, max_xy.y - delta))
            if not centers:
                suitable = True
            elif all([euclidean_distance(new_point, center) > 2*delta for center in centers]):
                suitable = True
            else:
                i += 1
                if i > 1000:
                    # raise ValueError("Cannot generate centers with given parameters. Try increasing delta or decreasing k or setting a bigger bounding box.")
                    return None
        centers.append(new_point)
        count += 1
    dataset = {}
    for center in centers:
        dataset[center] = []

    # generating points
    points_map = {}
    i = len(centers)
    while count < n:
        curr_center = centers[i % k]
        suitable = False
        while not suitable:
            new_point = XYPoint(random.gauss(curr_center.x, delta), random.gauss(curr_center.y, delta))
            if new_point.x > max_xy.x or new_point.x < min_xy.x or new_point.y > max_xy.y or new_point.y < min_xy.y:
                continue
            if new_point in points_map:
                continue
            if euclidean_distance(new_point, curr_center) <= delta:
                suitable = True
        dataset[curr_center].append(new_point)
        points_map[new_point] = True
        count += 1
        i += 1

    return dataset

class Clustering:
    """
    Clustering class.
    """

    def __init__(self, dissimilarity_function: Callable[[XYPoint, XYPoint], float]):
        """
        Clustering class constructor.

        :param dissimilarity_function: The dissimilarity function to use
        :type dissimilarity_function: Callable[[XYPoint, XYPoint], float]
        """
        self.dissimilarity_function = dissimilarity_function

    
    def __calc_dissimilarity(self, point1: XYPoint, point2: XYPoint) -> float:
        """
        Calculates the dissimilarity between two points.

        :param point1: The first point
        :type point1: XYPoint
        :param point2: The second point
        :type point2: XYPoint
        :return: The dissimilarity
        :rtype: float
        """
        return self.dissimilarity_function(point1, point2)


    def k_means(self, input_points: List[XYPoint], k: int, init_cetroids: List[XYPoint] = None) -> Tuple[Dict[XYPoint, List[XYPoint]], Dict]:
        """
        The k-means clustering algorithm.

        :param self: The Clustering object
        :type self: Clustering
        :param input_points: The points to cluster
        :type input_points: List[XYPoint]
        :param k: The number of clusters
        :type k: int
        :param init_cetroids: The initial centroids
        :type init_cetroids: List[XYPoint]
        :return: The clusters and the debug data
        :rtype: Tuple[Dict[XYPoint, List[XYPoint]], Dict]
        """

        # sanity checks
        if k < 1:
            raise ValueError("k must be greater than 0")
        if len(input_points) < k:
            raise ValueError("input_points must be greater than k")
        if init_cetroids and len(init_cetroids) != k:
            raise ValueError("init_cetroids must be of length k")
        if not input_points:
            raise ValueError("input_points must not be empty")

        # initialize centroids
        centroids = set()       # use set to avoid duplicates
        if not init_cetroids:
            while len(centroids) < k:
                centroids.add(input_points[random.randint(0, len(input_points) - 1)])
            centroids = list(centroids)     # convert back to list
        else:
            k = len(init_cetroids)
            centroids = init_cetroids

        clusters = {}

        # initialize clusters
        for centroid in centroids:
            clusters[centroid] = []
        
        n = len(input_points)
        iterations = 0
        assignments = 0
        cost_calculations = 0
        cluster_calculations = 0

        # assign each point to the closest centroid
        for point in input_points:
            closest_centroid = min(centroids, key=lambda c: self.__calc_dissimilarity(c, point))
            cost_calculations += k
            clusters[closest_centroid].append(point)
            assignments += 1

        # repeat until no change
        change_occured = True       # to enter the loop
        while (change_occured):
            iterations += 1
            change_occured = False

            # calculate new centroids
            new_centroids = []
            for prev_centroid, points in clusters.items():
                sum_x = 0
                sum_y = 0
                for point in points:
                    sum_x += point.x
                    sum_y += point.y

                mean_x = sum_x / float(len(points))
                mean_y = sum_y / float(len(points))
                new_centroids.append(XYPoint(mean_x, mean_y))
                cluster_calculations += 1

            # assign each point to the closest centroid
            new_clusters = {}
            for centroid in new_centroids:
                new_clusters[centroid] = []

            for prev_centroid, points in clusters.items():
                for point in points:
                    closest_centroid = min(new_centroids, key=lambda c: self.__calc_dissimilarity(c, point))
                    cost_calculations += k
                    if closest_centroid != prev_centroid:
                        change_occured = True
                    new_clusters[closest_centroid].append(point)
                    assignments += 1
            
            clusters = new_clusters
        
        debug_data = {}
        debug_data["iterations"] = iterations
        debug_data["assignments"] = assignments
        debug_data["cost_calculations"] = cost_calculations
        debug_data["cluster_calculations"] = cluster_calculations

        return clusters, debug_data
    
    def k_medoids(self, input_points: List[XYPoint], k: int, init_medoids: List[XYPoint] = None) -> Tuple[Dict[XYPoint, List[XYPoint]], Dict]:
        """
        The k-medoids clustering algorithm.

        :param self: The Clustering object
        :type self: Clustering
        :param input_points: The points to cluster
        :type input_points: List[XYPoint]
        :param k: The number of clusters
        :type k: int
        :param init_medoids: The initial medoids, defaults to None
        :type init_medoids: List[XYPoint], optional
        :return: The clusters and the debug data
        :rtype: Tuple[Dict[XYPoint, List[XYPoint]], Dict]
        """

        # sanity checks
        if k < 1:
            raise ValueError("k must be greater than 0")
        if len(input_points) < k:
            raise ValueError("input_points must be greater than k")
        if init_medoids and len(init_medoids) != k:
            raise ValueError("init_medoids must be of length k")
        if not input_points:
            raise ValueError("input_points must not be empty")
        
        # initialize medoids
        medoids = set()       # use set to avoid duplicates
        if not init_medoids:
            while len(medoids) < k:
                medoids.add(input_points[random.randint(0, len(input_points) - 1)])
            medoids = list(medoids)     # convert back to list
        else:
            medoids = init_medoids

        clusters = {}
        for medoid in medoids:
            clusters[medoid] = []

        n = len(input_points)
        iterations = 0
        assignments = 0
        cost_calculations = 0
        cluster_calculations = 0

        # assign each point to the closest medoid
        for point in input_points:
            closest_medoid = min(medoids, key=lambda m: self.__calc_dissimilarity(m, point))
            cost_calculations += k
            clusters[closest_medoid].append(point)
            assignments += 1

        # repeat until no change
        change_occured = True       # to enter the loop
        while change_occured:
            iterations += 1
            new_clusters = {}
            change_occured = False
            for ith_medoid, ith_cluster_points in clusters.items():
                n_cluster = len(ith_cluster_points)
                curr_cost = sum([self.__calc_dissimilarity(ith_medoid, point) for point in ith_cluster_points])
                cost_calculations += n_cluster
                cluster_calculations += 1
                new_medoid = None
                for point in ith_cluster_points:
                    new_cost = sum([self.__calc_dissimilarity(point, other_point) for other_point in ith_cluster_points if other_point != point])
                    new_cost += self.__calc_dissimilarity(point, ith_medoid)
                    cost_calculations += n_cluster
                    cluster_calculations += 1
                    if new_cost < curr_cost:
                        curr_cost = new_cost
                        new_medoid = point
                if new_medoid:
                    change_occured = True
                    new_clusters[new_medoid] = []
                    cluster_calculations += 1
                else:
                    new_clusters[ith_medoid] = []

            if change_occured:
                # assign each point to the closest new medoid
                for point in input_points:
                    closest_medoid = min(new_clusters.keys(), key=lambda m: self.__calc_dissimilarity(m, point))
                    cost_calculations += k
                    new_clusters[closest_medoid].append(point)
                    assignments += 1
                clusters = new_clusters

        debug_data = {}
        debug_data["iterations"] = iterations
        debug_data["assignments"] = assignments
        debug_data["cost_calculations"] = cost_calculations
        debug_data["cluster_calculations"] = cluster_calculations

        return clusters, debug_data
    

    def k_medoids_alternative(self, input_points: List[XYPoint], k: int) -> Tuple[Dict[XYPoint, List[XYPoint]], Dict]:
        # sanity checks
        if k < 1:
            raise ValueError("k must be greater than 0")
        if len(input_points) < k:
            raise ValueError("input_points must be greater than k")
        
        n = len(input_points)
        iterations = 0
        assignments = 0
        cost_calculations = 0
        cluster_calculations = 0
        vj_calculations = 0
        
        # calculate dissimilarity matrix
        dissimilarity_matrix = [[0]*n for _ in range(n)]  # Initialize the matrix
        cost_calculations += n
        for i in range(n):
            for j in range(i+1, n):  # start from i+1 to avoid the diagonal
                dissimilarity = self.__calc_dissimilarity(input_points[i], input_points[j])
                dissimilarity_matrix[i][j] = dissimilarity
                dissimilarity_matrix[j][i] = dissimilarity  # mirror the value
                cost_calculations += 1

        points_vj = [
            sum(
                [dissimilarity_matrix[i][j]/sum([dissimilarity_matrix[l][i] for l in range(n)]) 
                    for i in range(n)
                ]
            )
                for j in range(n)
                     ]
        vj_calculations += n**2

        # map points to their vj values
        points_vj = [(i, input_points[i], points_vj[i]) for i in range(n)]

        # sort points by their vj values
        points_vj.sort(key=lambda x: x[2])

        # select k medoids
        medoids = [(points_vj[i][0], points_vj[i][1]) for i in range(k)]
        
        # initialize clusters
        clusters = {}
        for medoid in medoids:
            clusters[medoid] = []

        # assign each point to the closest medoid
        for i in range(n):
            closest_medoid = min(medoids, key=lambda m: dissimilarity_matrix[m[0]][i])
            clusters[closest_medoid].append((i, input_points[i]))
            assignments += 1
        
        # repeat until no change
        change_occured = True       # to enter the loop
        while change_occured:
            iterations += 1
            change_occured = False
            new_clusters = {}
            for ith_medoid, ith_cluster_points in clusters.items():
                n_cluster = len(ith_cluster_points)
                curr_cost = sum([dissimilarity_matrix[ith_medoid[0]][point[0]] for point in ith_cluster_points])
                cost_calculations += 1      # as the distances are already calculated, only the sum is needed
                cluster_calculations += 1
                new_medoid = None
                for point in ith_cluster_points:
                    new_cost = sum([dissimilarity_matrix[point[0]][other_point[0]] for other_point in ith_cluster_points if other_point != point])
                    new_cost += dissimilarity_matrix[point[0]][ith_medoid[0]]
                    cost_calculations += 1     # as the distances are already calculated, only the sum is needed
                    cluster_calculations += 1
                    if new_cost < curr_cost:
                        curr_cost = new_cost
                        new_medoid = point
                if new_medoid:
                    change_occured = True
                    new_clusters[new_medoid] = []
                    cluster_calculations += 1
                else:
                    new_clusters[ith_medoid] = []

            if change_occured:
                # assign each point to the closest new medoid
                for i in range(n):
                    closest_medoid = min(new_clusters.keys(), key=lambda m: dissimilarity_matrix[m[0]][i])
                    new_clusters[closest_medoid].append((i, input_points[i]))
                    assignments += 1
                clusters = new_clusters

        debug_data = {}
        debug_data["iterations"] = iterations
        debug_data["assignments"] = assignments
        debug_data["cost_calculations"] = int(cost_calculations)
        debug_data["cluster_calculations"] = cluster_calculations
        debug_data["vj_calculations"] = vj_calculations

        # remove the indices from the clusters
        res_clusters = {}
        for medoid, points in clusters.items():
            res_clusters[medoid[1]] = [point[1] for point in points]

        return res_clusters, debug_data
    

    def k_medoids_alternative_timed(self, input_points: List[XYPoint], k: int) -> Tuple[Dict[XYPoint, List[XYPoint]], Dict]:
        # sanity checks
        if k < 1:
            raise ValueError("k must be greater than 0")
        if len(input_points) < k:
            raise ValueError("input_points must be greater than k")
        
        n = len(input_points)
        iterations = 0
        assignments = 0
        cost_calculations = 0
        cluster_calculations = 0
        vj_calculations = 0
        times = {}
        
        # calculate dissimilarity matrix
        timeit_start = timeit.default_timer()
        dissimilarity_matrix = [[0]*n for _ in range(n)]  # Initialize the matrix
        cost_calculations += n
        for i in range(n):
            for j in range(i+1, n):  # start from i+1 to avoid the diagonal
                dissimilarity = self.__calc_dissimilarity(input_points[i], input_points[j])
                dissimilarity_matrix[i][j] = dissimilarity
                dissimilarity_matrix[j][i] = dissimilarity  # mirror the value
                cost_calculations += 1
        timeit_stop = timeit.default_timer()
        times["dissimilarity_matrix"] = timeit_stop - timeit_start

        timeit_start = timeit.default_timer()
        points_vj = [
            sum(
                [dissimilarity_matrix[i][j]/sum([dissimilarity_matrix[l][i] for l in range(n)]) 
                    for i in range(n)
                ]
            )
                for j in range(n)
                     ]

        # map points to their vj values
        points_vj = [(i, input_points[i], points_vj[i]) for i in range(n)]

        # sort points by their vj values
        points_vj.sort(key=lambda x: x[2])
        timeit_stop = timeit.default_timer()
        times["points_vj"] = timeit_stop - timeit_start
        vj_calculations += n**2

        # select k medoids
        medoids = [(points_vj[i][0], points_vj[i][1]) for i in range(k)]
        
        timeit_start = timeit.default_timer()
        # initialize clusters
        clusters = {}
        for medoid in medoids:
            clusters[medoid] = []

        # assign each point to the closest medoid
        for i in range(n):
            closest_medoid = min(medoids, key=lambda m: dissimilarity_matrix[m[0]][i])
            clusters[closest_medoid].append((i, input_points[i]))
            assignments += 1
        
        # repeat until no change
        change_occured = True       # to enter the loop
        while change_occured:
            iterations += 1
            change_occured = False
            new_clusters = {}
            for ith_medoid, ith_cluster_points in clusters.items():
                n_cluster = len(ith_cluster_points)
                curr_cost = sum([dissimilarity_matrix[ith_medoid[0]][point[0]] for point in ith_cluster_points])
                cost_calculations += 1      # as the distances are already calculated, only the sum is needed
                cluster_calculations += 1
                new_medoid = None
                for point in ith_cluster_points:
                    new_cost = sum([dissimilarity_matrix[point[0]][other_point[0]] for other_point in ith_cluster_points if other_point != point])
                    new_cost += dissimilarity_matrix[point[0]][ith_medoid[0]]
                    cost_calculations += 1     # as the distances are already calculated, only the sum is needed
                    cluster_calculations += 1
                    if new_cost < curr_cost:
                        curr_cost = new_cost
                        new_medoid = point
                if new_medoid:
                    change_occured = True
                    new_clusters[new_medoid] = []
                    cluster_calculations += 1
                else:
                    new_clusters[ith_medoid] = []

            if change_occured:
                # assign each point to the closest new medoid
                for i in range(n):
                    closest_medoid = min(new_clusters.keys(), key=lambda m: dissimilarity_matrix[m[0]][i])
                    new_clusters[closest_medoid].append((i, input_points[i]))
                    assignments += 1
                clusters = new_clusters

        timeit_stop = timeit.default_timer()
        times["k_medoids"] = timeit_stop - timeit_start
        debug_data = {}
        debug_data["iterations"] = iterations
        debug_data["assignments"] = assignments
        debug_data["cost_calculations"] = int(cost_calculations)
        debug_data["cluster_calculations"] = cluster_calculations
        debug_data["vj_calculations"] = vj_calculations

        # remove the indices from the clusters
        res_clusters = {}
        for medoid, points in clusters.items():
            res_clusters[medoid[1]] = [point[1] for point in points]

        return res_clusters, debug_data, times

            
        


