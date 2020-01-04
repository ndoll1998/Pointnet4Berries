# import numpy
import numpy as np
from numpy import pi as PI
# import nearest neighbors
from sklearn.neighbors import NearestNeighbors
# import utils
from .utils import estimate_curvature_and_normals

def region_growing(points, normals, curvatures, theta_angle=PI/2, theta_curvature=0.3, radius=0.02, min_points=20):
    
    # build nearest neighbors
    tree = NearestNeighbors(algorithm='ball_tree', radius=radius, n_jobs=-1).fit(points)
    _, nearest_idx = tree.radius_neighbors(points)
    # delete nearest neighbor tree to free memory
    del tree

    # copy curvatures to manipulate
    curvatures = curvatures.copy()

    # map each point to a cluster by index
    cluster_mask = np.zeros(points.shape[0])
    cur_cluster_idx = 0

    # list of currently unassigned points
    A = set(range(points.shape[0]))

    # Region Growing Algorithm
    while len(A) > 0:
        # get first seed point and initialize Region and Seed lists
        i = np.argmin(curvatures)
        R, S = [i], [i]
        # remove from available list and make sure its not selected again
        A.difference_update([i])
        curvatures[i] = float('inf')

        # Grow region from seed
        while len(S) > 0:
            # get current seed
            i = S.pop(0)
            for j in nearest_idx[i]:
                # check if point is unassigned
                if j in A:
                    # compute angle between normals and handle rounding errors
                    angle = abs(normals[i, :] @ normals[j, :])
                    angle = min(max(angle, 0), 1)
                    # check if threshold is fulfilled
                    if np.arccos(angle) < theta_angle:
                        # check if seed is extended
                        if curvatures[j] < theta_curvature:
                            S.append(j)
                        # add to region and mark it as assigned
                        R.append(j)
                        A.difference_update([j])
                        curvatures[j] = float('inf')
                        
        # check for outlier
        if len(R) < min_points:
            # apply outlier cluster to points
            cluster_mask[R] = -1
        else:
            # apply cluster to points and increase current cluster index
            cluster_mask[R] = cur_cluster_idx
            cur_cluster_idx += 1

    return cluster_mask