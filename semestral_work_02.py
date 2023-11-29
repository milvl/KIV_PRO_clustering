import random
import csv
import os
import timeit
from pprint import pprint
import clustering as cl
from xypoint import XYPoint
from xypoint import euclidean_distance, manhattan_distance, cosine_similarity
import numpy as np
from sklearn.metrics import silhouette_score
from copy import deepcopy

# for charting
import matplotlib.pyplot as plt

def plot_dataset(dataset, label="Points"):
    # plot the dataset
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot points
    ax.plot([p.x for p in dataset], [p.y for p in dataset], 'ro', label=label)  # 'ro' means red dots

    # Add a legend
    ax.legend()

    # show the plot
    plt.show()

def plot_results(clusters, init_points=None, label="NA"):
    # now with colors
    colors = ['red', 'green', 'blue', 'yellow', 'orange']
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot points
    i = 0
    for _, points in clusters.items():
        ax.plot([p.x for p in points], [p.y for p in points], 'o', label=f'Body {i + 1}', color=colors[i])
        i += 1
    
    ax.plot([p.x for p in clusters.keys()], [p.y for p in clusters.keys()], 'x', label=f'{label}', color='black')

    if init_points:
            ax.plot([p.x for p in init_points], [p.y for p in init_points], 's', label=f'Initial {label}', color='purple')
    
    ax.legend()

    plt.title(label)
    plt.show()


def generate_sklearn_compatible_data(clusters_dataset, include_keys_as_data):
    # Flatten the dataset and generate labels
    mapping = {}
    i = 0
    for center, points in clusters_dataset.items():
        for point in points:
            mapping[point] = i
        if include_keys_as_data:
            mapping[center] = i
        i += 1

    # Generate dataset
    X = []
    labels = []
    for point, label in mapping.items():
        X.append([point.x, point.y])
        labels.append(label)

    return X, labels

def example():
    k = 4
    n = 100
    clusters_dataset = cl.create_dataset(n, k, None, XYPoint(-1000000000, -1000000000), XYPoint(1000000000, 1000000000))
    while not clusters_dataset:
        clusters_dataset = cl.create_dataset(n, k, None, XYPoint(-1000000000, -1000000000), XYPoint(1000000000, 1000000000))

    # start each function and plot it's results
    clustering = cl.Clustering(euclidean_distance)
    dataset = []
    for center, points in clusters_dataset.items():
        dataset.append(center)
        dataset.extend(points)
    init_points = random.sample(dataset, k=k)

    # plot dataset
    plot_dataset(dataset)
    # plot with actual clusters
    # append centers to dataset
    c_dataset = deepcopy(clusters_dataset)
    for center, points in c_dataset.items():
        c_dataset[center].append(center)
    plot_results(c_dataset, None, 'Origin point')

    # k-means
    c_clusters, debug_data = clustering.k_means(dataset, k, init_points)
    plot_results(c_clusters, init_points, 'k-means')

    # k-medoids
    c_clusters, debug_data = clustering.k_medoids(dataset, k, init_points)
    plot_results(c_clusters, init_points, 'k-medoids')

    # k-medoids alternative
    c_clusters, debug_data = clustering.k_medoids_alternative(dataset, k)
    plot_results(c_clusters, [], 'k-medoids alternative')


def meassuring():
    # ensure csv files (k_means.csv, k_medoids.csv and k_medoids_alt.csv) are present
    if not os.path.isfile('k_means.csv'):
        with open('k_means.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['dissimilarity_used', 'k', 'n', 'time_results', 'iterations', 'assignments', 'cost_calculations', 'cluster_calculations'])
    if not os.path.isfile('k_medoids.csv'):
        with open('k_medoids.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['dissimilarity_used', 'k', 'n', 'time_results', 'iterations', 'assignments', 'cost_calculations', 'cluster_calculations'])
    if not os.path.isfile('k_medoids_alt.csv'):
        with open('k_medoids_alt.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['dissimilarity_used', 'k', 'n', 'time_results', 'iterations', 'assignments', 'cost_calculations', 'cluster_calculations', 'vj_calculations'])

    distance_functions = [euclidean_distance, manhattan_distance, cosine_similarity]
    for i in range(len(distance_functions)):
        dissimilarity_used = ''
        if i == 0:
            dissimilarity_used = 'euclidean_distance'
        elif i == 1:
            dissimilarity_used = 'manhattan_distance'
        elif i == 2:
            dissimilarity_used = 'cosine_similarity'
             
        clustering = cl.Clustering(distance_functions[i])

        # for k-means
        print(f'k-means, {dissimilarity_used}')
        k = 2
        while k <= 7:
            n = 2500000
            time_result = 0
            # while duration < 2 min
            while time_result < 120:
                print(f'k={k}, n={n}')
                clusters_dataset = cl.create_dataset(n, k, None, XYPoint(-1000000000, -1000000000), XYPoint(1000000000, 1000000000))
                while not clusters_dataset:
                    clusters_dataset = cl.create_dataset(n, k, None, XYPoint(-1000000000, -1000000000), XYPoint(1000000000, 1000000000))
            
                dataset = []
                for center, points in clusters_dataset.items():
                    dataset.append(center)
                    dataset.extend(points)

                init_points = random.sample(dataset, k)

                # k-means
                timeit_start = timeit.default_timer()
                c_clusters, debug_data = clustering.k_means(dataset, k, init_points)
                timeit_stop = timeit.default_timer()
                time_result = timeit_stop - timeit_start
                
                # write to csv
                with open('k_means.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    res = [dissimilarity_used, k, n, time_result]
                    res.extend(list(debug_data.values()))
                    writer.writerow(res)
                print(f'time_result={time_result}')
                n += 2500000
            k += 1

        # for k-medoids
        print(f'k-medoids, {dissimilarity_used}')
        k = 2
        while k <= 7:
            n = 100
            time_result = 0
            # while duration < 2 min
            while time_result < 120:
                print(f'k={k}, n={n}')
                clusters_dataset = cl.create_dataset(n, k, None, XYPoint(-1000000000, -1000000000), XYPoint(1000000000, 1000000000))
                while not clusters_dataset:
                    clusters_dataset = cl.create_dataset(n, k, None, XYPoint(-1000000000, -1000000000), XYPoint(1000000000, 1000000000))
            
                dataset = []
                for center, points in clusters_dataset.items():
                    dataset.append(center)
                    dataset.extend(points)

                init_points = random.sample(dataset, k)

                # k-medoids
                timeit_start = timeit.default_timer()
                c_clusters, debug_data = clustering.k_medoids(dataset, k, init_points)
                timeit_stop = timeit.default_timer()
                time_result = timeit_stop - timeit_start
                
                # write to csv
                with open('k_medoids.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    res = [dissimilarity_used, k, n, time_result]
                    res.extend(list(debug_data.values()))
                    writer.writerow(res)
                print(f'time_result={time_result}')
                n *= 1.1
                n = int(n)
            k += 1

        # for k-medoids alternative
        print(f'k-medoids alternative, {dissimilarity_used}')
        k = 2
        while k <= 7:
            n = 100
            time_result = 0
            # while duration < 2 min
            while time_result < 120:
                clusters_dataset = cl.create_dataset(n, k, None, XYPoint(-1000000000, -1000000000), XYPoint(1000000000, 1000000000))
                while not clusters_dataset:
                    clusters_dataset = cl.create_dataset(n, k, None, XYPoint(-1000000000, -1000000000), XYPoint(1000000000, 1000000000))
            
                dataset = []
                for center, points in clusters_dataset.items():
                    dataset.append(center)
                    dataset.extend(points)

                # k-medoids alternative
                timeit_start = timeit.default_timer()
                c_clusters, debug_data = clustering.k_medoids_alternative(dataset, k)
                timeit_stop = timeit.default_timer()
                time_result = timeit_stop - timeit_start
                
                # write to csv
                with open('k_medoids_alt.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    res = [dissimilarity_used, k, n, time_result]
                    res.extend(list(debug_data.values()))
                    writer.writerow(res)
                print(f'time_result={time_result}')
                n *= 1.1
                n = int(n)
            k += 1


def comparison():
    dissimilarity_used = 'euclidean_distance'
    clustering = cl.Clustering(euclidean_distance)
    
    k = 5
    n = 100

    print(f'k={k}')
    print(f'n, k_means_time, k_medoids_time, k_medoids_alt_time')
    while n <= 1000:
        clusters_dataset = cl.create_dataset(n, k, None, XYPoint(-1000000000, -1000000000), XYPoint(1000000000, 1000000000))
        while not clusters_dataset:
            clusters_dataset = cl.create_dataset(n, k, None, XYPoint(-1000000000, -1000000000), XYPoint(1000000000, 1000000000))
    
        dataset = []
        for center, points in clusters_dataset.items():
            dataset.append(center)
            dataset.extend(points)

        init_points = random.sample(dataset, k=k)

        # k-means
        timeit_start = timeit.default_timer()
        c_clusters, debug_data = clustering.k_means(dataset, k, init_points)
        timeit_stop = timeit.default_timer()
        k_means_time = timeit_stop - timeit_start

        # k-medoids
        timeit_start = timeit.default_timer()
        c_clusters, debug_data = clustering.k_medoids(dataset, k, init_points)
        timeit_stop = timeit.default_timer()
        k_medoids_time = timeit_stop - timeit_start

        # k-medoids alternative
        timeit_start = timeit.default_timer()
        c_clusters, debug_data = clustering.k_medoids_alternative(dataset, k)
        timeit_stop = timeit.default_timer()
        k_medoids_alt_time = timeit_stop - timeit_start
        print(f'{n},{k_means_time},{k_medoids_time},{k_medoids_alt_time}')
        n += 100

def comparison_selective():
    dissimilarity_used = 'euclidean_distance'
    clustering = cl.Clustering(euclidean_distance)
    k = 5
    n = 100

    print(f'k={k}')
    print(f'n, (init_points_time,k_means_time),(init_points_time,k_medoids_time),(dissimilarity_matrix_time,points_vj_time,k_medoids_time)')
    while n <= 1000:
        clusters_dataset = cl.create_dataset(n, k, None, XYPoint(-1000000000, -1000000000), XYPoint(1000000000, 1000000000))
        while not clusters_dataset:
            clusters_dataset = cl.create_dataset(n, k, None, XYPoint(-1000000000, -1000000000), XYPoint(1000000000, 1000000000))
    
        dataset = []
        for center, points in clusters_dataset.items():
            dataset.append(center)
            dataset.extend(points)

        timeit_start = timeit.default_timer()
        init_points = random.sample(dataset, k=k)
        timeit_stop = timeit.default_timer()
        init_points_time = timeit_stop - timeit_start

        # k-means
        timeit_start = timeit.default_timer()
        c_clusters, debug_data = clustering.k_means(dataset, k, init_points)
        timeit_stop = timeit.default_timer()
        k_means_time = timeit_stop - timeit_start

        # k-medoids
        timeit_start = timeit.default_timer()
        c_clusters, debug_data = clustering.k_medoids(dataset, k, init_points)
        timeit_stop = timeit.default_timer()
        k_medoids_time = timeit_stop - timeit_start

        # k-medoids alternative
        c_clusters, debug_data, times = clustering.k_medoids_alternative_timed(dataset, k)
        print(f'{n},({init_points_time},{k_means_time}),({init_points_time},{k_medoids_time}),({times["dissimilarity_matrix"]},{times["points_vj"]},{times["k_medoids"]})')
        n += 100
     
def success_rate_test():
    dissimilarity_used = 'euclidean_distance'
    clustering = cl.Clustering(euclidean_distance)
    k = 5
    n = 100

    print(f'k={k}')
    print(f'n, k_means_score, k_medoids_score, k_medoids_alt_score')

    while n <= 1000:
        clusters_dataset = cl.create_dataset(n, k, None, XYPoint(-1000000000, -1000000000), XYPoint(1000000000, 1000000000))
        while not clusters_dataset:
            clusters_dataset = cl.create_dataset(n, k, None, XYPoint(-1000000000, -1000000000), XYPoint(1000000000, 1000000000))

        # Original dataset for silhouette evaluation
        sk_dataset, _ = generate_sklearn_compatible_data(clusters_dataset, True)
    
        dataset = []
        for center, points in clusters_dataset.items():
            dataset.append(center)
            dataset.extend(points)

        init_points = random.sample(dataset, k=k)

        # k-means
        c_clusters, debug_data = clustering.k_means(dataset, k, init_points)
        # convert to sklearn format (for k-means, centroids are not part of the clusters)
        _, sk_means = generate_sklearn_compatible_data(c_clusters, False)

        # k-medoids
        c_clusters, debug_data = clustering.k_medoids(dataset, k, init_points)
        # convert to sklearn format (for k-medoids, medoids are part of the clusters)
        _, sk_medoids = generate_sklearn_compatible_data(c_clusters, False)

        # k-medoids alternative
        c_clusters, debug_data = clustering.k_medoids_alternative(dataset, k)
        # convert to sklearn format (for k-medoids alternative, medoids are part of the clusters)
        _, sk_medoids_alt = generate_sklearn_compatible_data(c_clusters, False)

        # Calculate silhouette scores
        sk_means_score = silhouette_score(sk_dataset, sk_means)
        sk_medoids_score = silhouette_score(sk_dataset, sk_medoids)
        sk_medoids_alt_score = silhouette_score(sk_dataset, sk_medoids_alt)

        print(f'{n},{sk_means_score},{sk_medoids_score},{sk_medoids_alt_score}')
        n += 100 




if __name__ == "__main__":
    random.seed(hash('KIV/PRO'))
    example()
    comparison()
    comparison_selective()
    success_rate_test()
    meassuring()
