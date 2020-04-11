# import numpy
import numpy as np
# import open3d for 3d visualization
import open3d
# import utils
from utils.utils import normalize_pc, estimate_curvature_and_normals, get_points_in_bbox, group_points_by_octree
from utils.clustering import region_growing
from utils.data import class2color, get_voxel_subsamples, get_voxel_subsamples_equal_class_distribution
from utils.Visualizer import Visualizer
# import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
# import others
from math import ceil
from time import time
from tqdm import tqdm
from random import sample

def basic_visualization(pc):
    # get points and colors
    points = raw[:, :3]
    colors = raw[:, 3:6] / 255
    # get label-colors
    y = raw[:, -1:]
    classes = list(class2color.keys())
    get_color = lambda i: class2color[classes[int(i)]]
    label_colors = np.apply_along_axis(get_color, axis=-1, arr=y.reshape(-1, 1))
    # visualize
    vis = Visualizer(background_color=(1, 1, 1))
    vis.add_by_features(points, colors)
    vis.add_by_features(points, label_colors)
    vis.run()


def visualize_region_growing(pc):
    """ Visualization of region growing on a given set of points """

    pc_sub = pc[np.isin(pc[:, -1], [0, 1, 4]), :]
    pc_stem = pc[np.isin(pc[:, -1], [2, 5])]
    points, colors = pc_sub[:, :3], pc_sub[:, 3:6]
    # region growing
    # curvatures, normals = estimate_curvature_and_normals(points, n_neighbors=50)
    cluster_mask = region_growing(points, pc_sub[:, 6:9], pc_sub[:, 9:10], min_points=1000)

    # visualize clusters
    vis = Visualizer(background_color=(1, 1, 1))

    # color each region in a random color
    colors = np.empty_like(points)
    for i in np.unique(cluster_mask):
        colors[(cluster_mask==i), :] = np.random.uniform(0, 1, size=3)
        # vis.add_by_features(points[(cluster_mask==i)], colors[(cluster_mask==i)])
    # ignore outliers
    colors[(cluster_mask==-1), :] = 1

    vis.add_by_features(points, colors)
    vis.n -= 1
    vis.add_by_features(pc_stem[:, :3], np.zeros_like(pc_stem[:, :3]))
    vis.run()

def visualize_aligned_region_growing_clusters(pc, k=-1):
    """ Separate visualization of k normalized and aligned clusters created by region growing algorithm 
        where the colors for each point come from the distances to the nearest neighbors in a given radius
    """ 

    points, colors = pc[:, :3], pc[:, 3:6]
    # region growing
    curvatures, normals = estimate_curvature_and_normals(points, n_neighbors=250)
    cluster_mask = region_growing(points, normals, curvatures, min_points=1000)

    # get cluster indices of interest ignoreing outliers
    cluster_idx = np.unique(cluster_mask)
    cluster_idx = cluster_idx[(cluster_idx!=-1)]
    cluster_idx = cluster_idx[:k] if k > -1 else cluster_idx

    # create visualizer
    vis = Visualizer()
    # visualize each cluster
    for i in cluster_idx:
        # # get points of current cluster
        cluster = points[(cluster_mask==i)]
        cluster_colors = colors[(cluster_mask==i)]
        # # compute colors by nearest neighbors
        # tree = NearestNeighbors(algorithm='ball_tree', radius=0.03, n_jobs=-1).fit(normalize_pc(cluster))
        # distances, _ = tree.radius_neighbors(cluster)
        # # compute colors from distances
        # colors = np.asarray([[sum(ds)] for ds in distances]) + 0.5
        # colors = (colors / colors.max()).repeat(3, axis=-1)

        # principle component anaylsis
        pca = PCA(n_components=1).fit(cluster)
        # print(i, pca.explained_variance_, pca.explained_variance_ratio_)
        a = pca.components_[0].reshape(-1, 1)
        a /= np.linalg.norm(a)
        # priciple component must show away from origin such that the aligned custer/pointcloud is not upside down
        mean = cluster.mean(axis=0).reshape(-1, 1)
        b = mean / np.linalg.norm(mean)
        # compute angle between principle component and position of pointcloud and inverse direction if needed
        a *= 1 if (a.T @ b < 0) else -1
        # create rotation matrix to align principle component to (0, 1, 0)
        c = a + np.asarray([0, 1, 0]).reshape(-1, 1)    # b = (0, 1, 0).T
        R = 2 * (c @ c.T) / (c.T @ c) - np.eye(3)

        # add to visualizer
        # pc_orig = vis.add_by_features(cluster, colors)
        pc_rot = vis.add_by_features(cluster @ R.T, cluster_colors)
        # visualize princple components
        # origin_orig, origin_rot = np.asarray(pc_orig.points).mean(axis=0).reshape(-1, 1), np.asarray(pc_rot.points).mean(axis=0).reshape(-1, 1)
        # line_points = [a * 0.5 + origin_orig, -a * 0.5 + origin_orig, R @ a * 0.5 + origin_rot, -R @ a * 0.5 + origin_rot]
        # create lineset of principle components
        # lineset = open3d.geometry.LineSet()
        # lineset.points = open3d.utility.Vector3dVector(line_points)
        # lineset.lines = open3d.utility.Vector2iVector([[0, 1], [2, 3]])
        # lineset.colors = open3d.utility.Vector3dVector([(0, 0, 1), (0, 0, 1)])
        # add to visualizer
        # vis.add_geometry(lineset)

        # increase distance between clusters
        vis.n += 1

    vis.run(show_coordinate_frame=True)

def visualize_graph_partition(points, k=-1):
    """ Separate visualization of k normalized clusters created by region growing algorithm 
        where the colors show the best partition of the nearest neighbor graph in each cluster
    """ 
    # import graph and partition
    import networkx as nx
    import community

    # region growing
    curvatures, normals = estimate_curvature_and_normals(points, n_neighbors=50)
    cluster_mask = region_growing(points, normals, curvatures, min_points=100)

    # get cluster indices of interest ignoreing outliers
    cluster_idx = np.unique(cluster_mask)
    cluster_idx = cluster_idx[(cluster_idx!=-1)]
    cluster_idx = cluster_idx[:k] if k > -1 else cluster_idx

    # create visualizer
    vis = Visualizer(background=(1, 1, 1))
    # visualize each cluster
    for i in cluster_idx:
        # get points of current cluster
        cluster = normalize_pc(points[(cluster_mask==i)])
        # build ajacency matrix for neighbor graph
        tree = NearestNeighbors(algorithm='ball_tree', radius=0.03, n_jobs=-1).fit(cluster)
        graph_matrix = tree.radius_neighbors_graph(cluster, mode='distance')
        # build graph from sparse adjacency matrix and compute best partition
        G = nx.from_scipy_sparse_matrix(graph_matrix)
        partition = community.best_partition(G)
        # color each partition in a random color
        community_colors, colors = {}, np.empty_like(cluster)
        for i, c in partition.items():
            # get random color
            if c not in community_colors:
                community_colors[c] = np.random.uniform(0, 1, size=3)
            # set color
            colors[i, :] = community_colors[c]

        # add to visualizer
        vis.add_by_features(cluster, colors)

    # show
    vis.run()

def visualize_pointcloud_breeding(pc):

    # get normals and curvature
    normals, curvature = pc[:, 6:9], pc[:, 9]
    
    # augment from each point
    points = pc[:, :3]
    # generate one vector on each plane spanned by a 
    # point and its normal in the pointcloud
    r = np.random.uniform(-1, 1, size=points.shape)
    n = np.cross(normals, r)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    # create new points
    d = curvature / curvature.max() * 1e-2
    new_points = points + n * d.reshape(-1, 1)

    # find the nearest neighbors of each new point
    tree = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', n_jobs=-1).fit(points)
    nearest_idx = tree.kneighbors(new_points, return_distance=False)
    # interpolate color from the n nearest points
    colors = pc[nearest_idx, 3:6].mean(axis=1)

    # stack all together
    full_points = np.concatenate((points, new_points), axis=0)
    full_colors = np.concatenate((pc[:, 3:6], colors), axis=0)

    # show
    vis = Visualizer()
    vis.add_by_features(points, pc[:, 3:6]/255)
    vis.add_by_features(new_points, colors/255)
    vis.add_by_features(full_points, full_colors/255)
    vis.run(True)    

def visualize_aligned_region_growing_and_breeding_clusters(pc, k=-1):
    """ Separate visualization of k normalized, aligned and breeded 
        clusters created by region growing algorithm 
    """ 

    # region growing
    points, curvatures, normals = pc[:, :3], pc[:, 9], pc[:, 6:9]
    cluster_mask = region_growing(points, normals, curvatures, min_points=1000)

    # get cluster indices of interest ignoreing outliers
    cluster_idx = np.unique(cluster_mask)
    cluster_idx = cluster_idx[(cluster_idx!=-1)]
    cluster_idx = cluster_idx[:k] if k > -1 else cluster_idx

    # create visualizer
    vis = Visualizer()
    # visualize each cluster
    for i in cluster_idx:
        # get points of current cluster
        mask = (cluster_mask==i)
        cluster = points[mask]
        # make colors from classes
        get_color = lambda i: list(class2color.values())[int(i)]
        colors = np.apply_along_axis(get_color, axis=-1, arr=pc[mask, -1:])

        # principle component anaylsis
        pca = PCA(n_components=1).fit(cluster)
        # print(i, pca.explained_variance_, pca.explained_variance_ratio_)
        a = pca.components_[0].reshape(-1, 1)
        a /= np.linalg.norm(a)
        # priciple component must show away from origin such that the aligned custer/pointcloud is not upside down
        mean = cluster.mean(axis=0).reshape(-1, 1)
        b = mean / np.linalg.norm(mean)
        # compute angle between principle component and position of pointcloud and inverse direction if needed
        a *= 1 if (a.T @ b < 0) else -1
        # create rotation matrix to align principle component to (0, 1, 0)
        c = a + np.asarray([0, 1, 0]).reshape(-1, 1)    # b = (0, 1, 0).T
        R = 2 * (c @ c.T) / (c.T @ c) - np.eye(3)

        # rotate points
        rotated_points = cluster @ R.T
        # normalize points
        rotated_points = normalize_pc(rotated_points)

        # get normals and curvature of current cluster
        cur_normals, cur_curvature = normals[mask], curvatures[mask]

        l = max(ceil(10_000/cluster.shape[0]) - 1, 0)
        # generate one vector on each plane spanned by a 
        # point and its normal in the pointcloud
        r = np.random.uniform(-1, 1, size=(cluster.shape[0] * l, 3))
        n = np.cross(cur_normals.repeat(l, axis=0), r)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        # create new points
        d = np.random.uniform(-5e-2, 5e-2, size=n.shape[0])
        new_points = rotated_points.repeat(l, axis=0) + n * d.reshape(-1, 1)

        # stack points together
        all_points = np.concatenate((rotated_points, new_points), axis=0)
        all_colors = colors.repeat(l+1, axis=0)

        # visualize
        vis.add_by_features(rotated_points, colors/255, normalize=False)
        vis.add_by_features(all_points, all_colors/255, normalize=False)

        # increase distance between pointclouds
        vis.n += 1

    vis.run(show_coordinate_frame=True)

def voxel_down_sample_random(pc, voxel_grid_size=0.2, n_points=35_000):

    # get points
    points = pc[:, :3]
    colors = np.ones_like(points)

    # get boundings of points
    max_ = np.max(points, axis=0)
    min_ = np.min(points, axis=0)
    # move bounding box anchor to origin
    bbox = max_ - min_
    # compute number of voxels in each dimesion
    n_voxels = np.ceil(bbox/voxel_grid_size).astype(np.int32)

    voxel_points = []
    # loop over all voxels
    for index in tqdm(np.ndindex(*n_voxels), total=np.product(n_voxels)):
        # build anchors of current voxels
        anchorA = np.asarray(index) * voxel_grid_size + min_
        anchorB = anchorA + voxel_grid_size
        # get points in current voxel
        point_idx = get_points_in_bbox(points, anchorA, anchorB)
        if len(point_idx) > 0:
            voxel_points.append(point_idx)

    sampled_point_idx = []
    # compute weight of first voxel
    weights = np.asarray([len(idx) / points.shape[0] for idx in voxel_points])
    # weights = 1 - np.asarray([len(idx) / points.shape[0] for idx in voxel_points])
    # weight = weights[0] / np.sum(weights)
    # sample random points from each voxel
    for i, idx in enumerate(voxel_points):
        # get number of points to sample from current voxel
        n_points_from_current = min(ceil(weights[i] * n_points), len(idx))
        # sample points random
        sampled_point_idx += sample(list(idx), n_points_from_current)

        # if i+1 < len(voxel_points):
        #     # update weight
        #     weights = 1 - np.asarray([len(idx) / (points.shape[0] - len(sampled_point_idx)) for idx in voxel_points[i+1:]])
        #     weight = weights[0] / np.sum(weights)

        # color points of current voxel
        colors[idx] = np.random.uniform(0, 1, size=3)

    print(len(sampled_point_idx), n_points, n_voxels)

    # get sampled points
    sampled_point_idx = sample(sampled_point_idx, n_points)
    sampled_points = points[sampled_point_idx, :]
    sampled_colors = colors[sampled_point_idx, :]

    random_point_idx = sample(range(points.shape[0]), n_points)
    random_points = points[random_point_idx, :]
    random_colors = colors[random_point_idx, :]

    # visualize
    vis = Visualizer(background_color=(1, 1, 1))
    vis.add_by_features(points, colors)
    # vis.add_by_features(random_points, random_colors)
    vis.add_by_features(sampled_points, sampled_colors)
    vis.run()

def voxel_down_sample_equally_distributed(pc, n_points):
    
    # split pointcloud in classes
    pc_per_class = [pc[pc[:,-1] == i] for i in np.unique(pc[:, -1])]
    # get number of points to sample from each class
    points_per_class = np.array([n_points // len(pc_per_class)] * len(pc_per_class))
    # handle sample larger than population
    overflow = np.array([max(0, n - pc_c.shape[0]) for pc_c, n in zip(pc_per_class, points_per_class)])    
    addition = np.zeros_like(points_per_class).astype(int)
    addition[overflow == 0] += overflow.sum() // (overflow == 0).sum()
    points_per_class = points_per_class - overflow + addition
    # sample from all classes
    sample = [get_voxel_subsamples(pc_c, n, 1)[0] for pc_c, n in zip(pc_per_class, points_per_class)]
    sample = np.concatenate(sample, axis=0)

    print(sum(points_per_class))
    print(sample.shape[0])
    print([(sample[:, -1] == i).sum() for i in np.unique(pc[:, -1])])

def voxel_down_sample_octree(pc, n_points):

    points = pc[:, :3]
    print(points.shape[0])
    voxels = group_points_by_octree(points, 1_000)

    vis = Visualizer(background_color=(1, 1, 1))

    colors = np.empty_like(points)
    for vox in voxels:
        colors[vox, :] = np.random.uniform(0, 1, size=3)

    vis.add_by_features(points, colors)

    samples_points = []
    # get the number of points selected from each voxel
    weights = np.asarray([len(voxel)/points.shape[0] for voxel in voxels])
    points_per_voxel = np.ceil(weights * n_points).astype(np.int32)
    # select random points from each voxel and add to list
    idx = sum([sample(list(v), min(n, len(v))) for v, n in zip(voxels, points_per_voxel)], [])

    vis.add_by_features(points[idx, :], colors[idx, :])

    vis.run()



if __name__ == "__main__":

    # load pointcloud
    raw = np.loadtxt("C:/Users/Nicla/Google Drive/P4B/data/Segmentation/CB_2E.feats")
    # rawB = np.loadtxt("C:/Users/Nicla/Google Drive/P4B/data/Classification/D_03.xyzrgb")
    # normalize the pointcloud
    raw[:, :3] = normalize_pc(raw[:, :3])
    voxel_down_sample_octree(raw, n_points=70_000)

    # basic_visualization(raw)


    # downsampling
    # voxel_down_sample_random(raw)

    # get all points not in stem
    # raw = raw[np.isin(raw[:, -1], [0, 1, 4]), :]

    # breeding pointcloud
    # visualize_region_growing(raw)

    # segmentating pointcloud
    # points = raw[:, :3]
    # visualize_region_growing(points)
    # visualize_aligned_region_growing_clusters(points, k=5)
    # visualize_graph_partition(points, k=1)
