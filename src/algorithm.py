from tqdm import tqdm
import numpy as np
from copy import deepcopy
from sklearn import mixture


def kmean(data, k, steps, problem_id=1):
    data_dims = len(data)
    data_size = len(data[0])

    means = [[] for _ in range(0, k)]
    cluster_idx = np.zeros((data_size,), dtype=np.int)

    # init centers
    for dim_d in data:
        max_v, min_v = np.max(dim_d), np.min(dim_d)
        mean_v = np.mean(dim_d)
        var_v = np.var(dim_d)
        print(max_v, min_v)
        for kdx in range(k):
            means[kdx].append(min_v + np.random.random() * (max_v - min_v))

    # main loop
    for sdx in range(0, steps):
        print(sdx)
        clusters = [[[] for _ in range(0, data_dims)] for _ in range(k)]
        print(means)

        # for each sample
        for ddx in range(data_size):
            l2_dists = []
            sample = np.asarray([dim_d[ddx] for dim_d in data])
            for mean in means:
                if problem_id == 1:
                    l2_dist = np.sum([(x1 - x2) ** 2 for x1, x2 in zip(sample, np.asarray(mean))])
                    l2_dists.append(l2_dist)
                elif problem_id == 2:
                    l2_dist1 = np.sum([(x1 - x2) ** 2 for x1, x2 in zip(sample[:2], np.asarray(mean)[:2])])
                    l2_dist2 = np.sum([(x1 - x2) ** 2 for x1, x2 in zip(sample[2:], np.asarray(mean)[2:])])
                    l2_dists.append(l2_dist1 + 0.05 * l2_dist2)

            min_dist = int(np.argmin(l2_dists))
            for dim in range(0, data_dims):
                clusters[min_dist][dim].append(data[dim][ddx])
            cluster_idx[ddx] = min_dist
        # update means
        for cdx, cluster in enumerate(clusters):
            for idx, dim in enumerate(cluster):
                if len(dim) != 0:
                    means[cdx][idx] = np.mean(np.array(dim))

        visulization(means, clusters)

    return np.asarray(means), cluster_idx


def mean_shift(data, bandwidth, steps, problem_id=1, cluster_threshold=1e-1, visual=True):
    data_dims = len(data)
    data_size = len(data[0])

    min_distance = 1e-1
    max_min_dist = 1

    need_shift = [True] * data_size

    # store shift data
    mean_shift_data = deepcopy(data)

    # iteration number
    for sdx in range(0, steps):
    # while max_min_dist > min_distance:
        max_min_dist = 0
        # handel each sample
        for ddx in range(0, data_size):
            # if not need_shift[ddx]:
            #     continue

            sample = np.asarray([dim_d[ddx] for dim_d in mean_shift_data])

            shift = [0.0] * data_dims
            scale = 0.0
            for idx in range(0, data_size):
                tmp_sample = np.asarray([dim_d[idx] for dim_d in data])
                if problem_id == 1:
                    distance = np.linalg.norm(tmp_sample - sample)
                    weight = (1 / (bandwidth * np.sqrt(2 * np.pi))) \
                                * np.exp(-0.5 * (distance / bandwidth) ** 2)

                elif problem_id == 2:
                    distance1 = np.linalg.norm(tmp_sample[:2] - sample[:2])
                    distance2 = np.linalg.norm(tmp_sample[2:] - sample[2:])
                    weight = (1 / (bandwidth[0] * bandwidth[1] * np.sqrt(2 * np.pi))) \
                                * np.exp(-0.5 * ((distance1 / bandwidth[0]) ** 2 - (distance2 / bandwidth[1]) ** 2))

                for jdx in range(0, data_dims):
                    shift[jdx] += tmp_sample[jdx] * weight
                scale += weight

            shift = np.asarray(shift) / scale
            old_shift = np.asarray([mean_shift_data[d][ddx] for d in range(0, data_dims)])
            move_dist = np.linalg.norm(shift - old_shift)

            if move_dist < min_distance:
                need_shift[ddx] = False
            if move_dist > max_min_dist:
                max_min_dist = move_dist
            print(move_dist)
            for d in range(0, data_dims):
                mean_shift_data[d][ddx] = shift[d]

    # cluster points
    cluster_idx = []
    c_number = 0
    cluster_centers = []

    for ddx in range(0, data_size):
        sample = np.asarray([dim_d[ddx] for dim_d in mean_shift_data])

        if len(cluster_idx) == 0:
            cluster_idx.append(c_number)
            cluster_centers.append(sample)
            c_number += 1
        else:
            for cdx, center in enumerate(cluster_centers):
                dist = np.linalg.norm(sample - center)
                if dist < cluster_threshold:
                    print(dist)
                    cluster_idx.append(cdx)
                    break

            if len(cluster_idx) < ddx + 1:
                cluster_idx.append(c_number)
                cluster_centers.append(sample)
                c_number += 1

    print(c_number)

    if visual:
        clusters = [[[] for _ in range(0, data_dims)] for _ in range(c_number)]
        for ddx in range(0, data_size):
            for d in range(0, data_dims):
                clusters[cluster_idx[ddx]][d].append(data[d][ddx])
        visulization(None, clusters)

    return np.asarray(mean_shift_data), np.asarray(cluster_idx)


def em_gmm(data, k, problem_id=1, visual=True):
    data_size = len(data[0])
    data_dims = len(data)
    dataset = []
    for i in range(0, data_size):
        dataset.append([data[0][i], data[1][i]])

    clf = mixture.GaussianMixture(n_components=k, covariance_type='full')
    clf.fit(np.array(dataset))

    label = []
    for d in dataset:
        label.append(clf.predict([d])[0])
    print(label)

    if visual:
        clusters = [[[] for _ in range(0, data_dims)] for _ in range(k)]
        for i, l in enumerate(label):
            for dim in range(0, data_dims):
                clusters[l][dim].append(data[dim][i])

        visulization(None, clusters)
    return np.asarray(label), np.asarray(label)


color_map = {
    -1: 'indianred',
    0: 'orange',
    1: 'khaki',
    2: 'lightgreen',
    3: 'paleturquoise',
    4: 'dodgerblue',
    5: 'lightsteelblue',
    6: 'slategray',
    7: 'mediumpurple',
    8: 'hotpink',
    9: 'silver',
}



import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def visulization(means, clusters):
    plt.clf()
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    for cdx, cluster in enumerate(clusters):
        if means is not None:
            plt.scatter([means[cdx][0]], [means[cdx][1]], c=color_map[cdx], marker='x')
        plt.scatter(cluster[0], cluster[1], c=color_map[cdx], s=10)
    plt.show()


if __name__ == '__main__':
    from src.dataloader import mat_data_loader

    data = mat_data_loader('../data/PA2-cluster-data/cluster_data.mat', 1)

    # kmean(data['dataA_X'], 4, 100)
    # kmean(data['dataB_X'], 4, 100)
    # kmean(data['dataC_X'], 4, 100)

    # mean_shift(data['dataA_X'], 2, 20)
    # mean_shift(data['dataB_X'], 1.5, 50)
    # mean_shift(data['dataC_X'], 2.5, 20)

    # em_gmm(data['dataA_X'], 4)
    # em_gmm(data['dataB_X'], 4)
    # em_gmm(data['dataC_X'], 4)





