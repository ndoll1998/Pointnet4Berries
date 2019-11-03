# import torch
import torch

def square_distance(src, dst):
    # get shapes
    (B, N, _), (_, M, _) = src.shape, dst.shape
    # compute distace from each src-point to each dst-point
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(npoint, points):
    # get device and shape of given data
    device, (B, N, _) = points.device, points.shape
    # create tensors
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # others
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # sample by maximum distance
    for i in range(npoint):
        # set centroid to farthest point
        centroids[:, i] = farthest
        # compute distance
        centroid = points[batch_indices, farthest, :].view(B, 1, -1)
        dist = torch.sum((points - centroid) ** 2, -1)
        # update distances to closest cetroid
        mask = dist < distance        
        distance[mask] = dist[mask]
        # get next farthest point
        farthest = torch.max(distance, -1)[1]
    
    return centroids


def query_ball_point(radius, centroids, points, nsample):
    # get decvice and dimensions
    device, (B, N, C), (_, S, _) = points.device, points.shape, centroids.shape
    # create initial grouping
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    
    # compute distances from each centroid to each point
    sqrdists = square_distance(centroids, points)
    # group all points with bigger radius than maximum
    group_idx[sqrdists > radius ** 2] = N - 1
    # group by distance
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # apply grouping
    mask = group_idx == N - 1
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(points, feats, n_clusters, radius, n_sample, returnfps=False):

    points = points.transpose(1, 2)
    # get shape
    device, (B, N, C) = points.device, points.shape
    # others
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # sample centroids
    centroid_idx = farthest_point_sample(n_clusters, points)
    centroids = points[batch_indices.unsqueeze(-1), centroid_idx]
    # group points
    idx = query_ball_point(radius, centroids, points, n_sample)
    grouped_points = points.unsqueeze(1)[batch_indices.view(-1, 1, 1), [0], idx]
    # set group origin to be centroid
    y = grouped_points - centroids.view(B, n_clusters, 1, C)
    # concatenate with features
    if feats.size(1) > 0:
        feats = feats.transpose(1, 2)
        grouped_feats = feats.unsqueeze(1)[batch_indices.view(-1, 1, 1), [0], idx]
        y = torch.cat([y, grouped_feats], dim=-1)
    # return 
    if returnfps:
        return centroids, y, grouped_points, centroid_idx
    else:
        return centroids, y


def interpolate(y_points, x_points, y_feats, x_feats):
    # transpose
    y_points, x_points = y_points.transpose(1, 2), x_points.transpose(1, 2)
    # get device and dimensions
    device, (B, N, dim), = x_points.device, x_points.shape
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # compute distances and ignore distances of 0
    D = square_distance(x_points, y_points)
    D += 1e-10  # D[D == 0] = float('inf')
    # sort distances to interpolate from the closest
    D, idx = D.sort(dim=-1)
    # get interpolation points
    D, idx = D[:, :, :dim], idx[:, :, :dim]
    # compute interploation weights
    weight = 1.0 / D
    weight = weight / weight.sum(dim=-1).view(B, N, 1)
    # compute interpolation points
    i_points = y_feats.transpose(1, 2).unsqueeze(1)[batch_indices.view(-1, 1, 1), [0], idx]
    i_points = (i_points * weight.unsqueeze(-1)).sum(dim=2)
    # concatenate with x-feats
    i_points = torch.cat([x_feats.transpose(1, 2), i_points], dim=-1)
    # return
    return i_points


# *** SCRIPT ***

if __name__ == '__main__':

    import numpy as np

    # create random points
    a = np.random.randint(0, 9, size=(1, 50, 2))
    a = torch.from_numpy(a).float()
    
    sample_and_group(a, 5, 3, 7, dim=2)
