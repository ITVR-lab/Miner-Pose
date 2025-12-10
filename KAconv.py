import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce


class KMeansCluster(nn.Module):
    def __init__(self, cluster_num, max_iter=20, tol=0.005):
        super().__init__()
        self.cluster_num = cluster_num
        self.max_iter = max_iter
        self.tol = tol
        
    def forward(self, features):
        batch_size, num_samples, feature_dim = features.shape
        device = features.device
        
        cluster_indices = []
        for b in range(batch_size):
            sample = features[b]
            
            indices = torch.randperm(num_samples, device=device)[:self.cluster_num]
            centers = sample[indices]
            
            for _ in range(self.max_iter):
                distances = torch.cdist(sample, centers)
                assignments = torch.argmin(distances, dim=1)
                
                new_centers = torch.zeros_like(centers)
                for k in range(self.cluster_num):
                    mask = (assignments == k)
                    if mask.sum() > 0:
                        new_centers[k] = sample[mask].mean(dim=0)
                    else:
                        new_centers[k] = centers[k]
                
                center_shift = torch.norm(new_centers - centers, dim=1).sum()
                centers = new_centers
                
                if center_shift < self.tol * num_samples:
                    break
            
            cluster_indices.append(assignments)
        
        return torch.stack(cluster_indices, dim=0)


def compute_cluster_centers(patches, cluster_indices, cluster_num):
    batch_size, num_patches, feature_dim = patches.shape
    device = patches.device
    
    cluster_indices_expanded = repeat(cluster_indices, 'b p -> b p f', f=feature_dim)
    
    cluster_sums = torch.zeros(batch_size, cluster_num, feature_dim, device=device).scatter_add_(
        dim=1,
        index=cluster_indices_expanded,
        src=patches
    )
    
    cluster_counts = torch.zeros(batch_size, cluster_num, device=device).scatter_add_(
        dim=1,
        index=cluster_indices,
        src=torch.ones(batch_size, num_patches, device=device)
    ).unsqueeze(2).clamp(min=1)
    
    cluster_centers = cluster_sums / cluster_counts
    
    return cluster_centers


def dispatch_and_permute(indices, total_clusters):
    counts = torch.bincount(indices, minlength=total_clusters)
    cluster_sizes, cluster_perm = counts.sort(descending=True)
    offsets = F.pad(torch.cumsum(cluster_sizes, dim=0), (1, 0))
    
    index_perm = torch.zeros_like(indices)
    for i in range(total_clusters):
        mask = (indices == cluster_perm[i])
        index_perm[mask] = i
    
    return index_perm, offsets, cluster_perm


def apply_cluster_conv(patches, indices, weights, biases=None):
    batch_size, num_patches, _ = patches.shape
    num_clusters = weights.shape[1]
    
    patches_flat = rearrange(patches, 'b p f -> (b p) f')
    weights_flat = rearrange(weights, 'b k f c -> (b k) f c')
    indices_flat = indices + torch.arange(batch_size, device=indices.device).view(-1, 1) * num_clusters
    indices_flat = rearrange(indices_flat, 'b p -> (b p)')
    
    if biases is not None:
        biases_flat = rearrange(biases, 'b k c -> (b k) c')
    
    index_perm, offsets, cluster_perm = dispatch_and_permute(indices_flat, batch_size * num_clusters)
    
    patches_sorted = patches_flat[torch.argsort(index_perm)]
    
    output_sorted = torch.zeros(patches_flat.shape[0], weights.shape[-1], device=patches.device, dtype=patches.dtype)
    
    for i in range(len(offsets) - 1):
        start, end = offsets[i], offsets[i + 1]
        if start >= end:
            continue
        
        w = weights_flat[cluster_perm[i]]
        output_sorted[start:end] = torch.matmul(patches_sorted[start:end], w)
        
        if biases is not None:
            output_sorted[start:end] += biases_flat[cluster_perm[i]]
    
    inverse_perm = torch.argsort(torch.argsort(index_perm))
    output = output_sorted[inverse_perm]
    
    return rearrange(output, '(b p) c -> b p c', b=batch_size)


class KAconv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 cluster_num=32,
                 mlp_hidden_dim=64,
                 num_base_kernels=None,
                 bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.cluster_num = cluster_num
        self.kernel_area = kernel_size ** 2
        self.use_bias = bias
        
        if num_base_kernels is None:
            num_base_kernels = cluster_num
        self.num_base_kernels = num_base_kernels
        
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        
        self.kmeans = KMeansCluster(cluster_num)
        
        self.kernel_generator = nn.Sequential(
            nn.Linear(in_channels * self.kernel_area, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, num_base_kernels),
            nn.Softmax(dim=-1)
        )
        
        self.base_kernels = nn.Parameter(
            torch.randn(num_base_kernels, in_channels * self.kernel_area, out_channels)
        )
        nn.init.kaiming_normal_(self.base_kernels, nonlinearity='relu')
        
        if self.use_bias:
            self.bias_generator = nn.Sequential(
                nn.Linear(in_channels * self.kernel_area, mlp_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_dim, out_channels)
            )
    
    def generate_adaptive_kernels(self, cluster_centers):
        attention_weights = self.kernel_generator(cluster_centers)
        attention_weights = rearrange(attention_weights, 'b k n -> b k n 1 1')
        
        adaptive_kernels = torch.sum(
            attention_weights * self.base_kernels.unsqueeze(0).unsqueeze(0),
            dim=2
        )
        
        return adaptive_kernels
    
    def generate_adaptive_biases(self, cluster_centers):
        if self.use_bias:
            return self.bias_generator(cluster_centers)
        return None
    
    def extract_local_features(self, x, patches):
        observation_vectors = reduce(
            patches,
            'b p (c area) -> b p c',
            'mean',
            c=self.in_channels,
            area=self.kernel_area
        )
        return observation_vectors
    
    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        
        patches = self.unfold(x)
        patches = rearrange(
            patches,
            'b (c area) (h w) -> b (h w) (c area)',
            c=in_channels,
            area=self.kernel_area,
            h=height,
            w=width
        )
        
        observation_features = self.extract_local_features(x, patches)
        
        cluster_indices = self.kmeans(observation_features)
        
        cluster_centers = compute_cluster_centers(patches, cluster_indices, self.cluster_num)
        
        adaptive_kernels = self.generate_adaptive_kernels(cluster_centers)
        adaptive_biases = self.generate_adaptive_biases(cluster_centers)
        
        output = apply_cluster_conv(patches, cluster_indices, adaptive_kernels, adaptive_biases)
        
        output = rearrange(output, 'b (h w) c -> b c h w', h=height, w=width)
        
        return output, cluster_indices
