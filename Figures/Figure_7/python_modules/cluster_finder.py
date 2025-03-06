import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

class ClusterFinder:
    
    def __init__(self,raw_heat_map: np.array):
        
        if raw_heat_map is not None:
            self.heatmap = raw_heat_map
        else:
            raise ValueError("You must provide a np.array representing a heatmap")
        
        self.high_values_auto = None
        self.labels_auto = None
        self.percentile_threshold = None
        self.raw_cluster_points = None
        print("Successfully initialized")
        
    def get_values_by_threshold(self, threshold: int):
        # Calculate the automatic threshold as the 95th percentile of the heatmap values
        self.percentile_threshold = np.percentile(self.heatmap, threshold)
        # Extract high-value points based on the calculated threshold
        self.high_values_auto = np.argwhere(self.heatmap > self.percentile_threshold)

    def get_dbscan(self, threshold = 90):
        self.get_values_by_threshold(threshold)
        # Perform DBSCAN clustering on these points
        dbscan_auto = DBSCAN(eps=20, min_samples=10).fit(self.high_values_auto)
        self.labels_auto = dbscan_auto.labels_

    def get_visualization(self, save_path: str = None):
        # Visualize the heatmap with automatically determined threshold clusters
        plt.figure(figsize=(12, 8))
        plt.imshow(self.heatmap, cmap='hot', extent=[0, 2560, 0, 1920], origin='lower')
        plt.colorbar(label="Heat Intensity")

        # Overlay clusters with polygons
        for label in set(self.labels_auto):
            if label == -1:
                continue  # Skip noise points
            cluster_points = self.high_values_auto[self.labels_auto == label]
            self.raw_cluster_points = cluster_points
            hull = plt.Polygon(cluster_points, edgecolor='blue', fill=False, linewidth=1.5)
            plt.gca().add_patch(hull)

        plt.title(f"Heatmap with Clusters (Threshold = {self.percentile_threshold:.2f})")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        
        if save_path:
            # Save the figure to the specified path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print("Image succesfully save")
        plt.show()
        
    def calculate_correlation(self):
        if self.high_values_auto is None or self.labels_auto is None:
            raise ValueError("Run 'get_dbscan' before calculating correlation.")

        # Check if the cluster points are valid heatmap indices
        raw_cluster_points = self.high_values_auto
        try:
            # Extract heatmap values at the cluster points
            cluster_values = self.heatmap[raw_cluster_points[:, 0], raw_cluster_points[:, 1]]
        except IndexError:
            # If raw_cluster_points do not match heatmap indices, scale them back
            print("Scaling raw_cluster_points based on extent.")
            x_scale = self.heatmap.shape[1] / 2560  # Width scale
            y_scale = self.heatmap.shape[0] / 1920  # Height scale
            scaled_points = (raw_cluster_points * [y_scale, x_scale]).astype(int)
            cluster_values = self.heatmap[scaled_points[:, 0], scaled_points[:, 1]]

        # Extract the brighter zones of the heatmap based on threshold
        bright_mask = (self.heatmap > self.percentile_threshold)
        bright_values = self.heatmap[bright_mask]

        # Calculate Pearson correlation
        corr, _ = pearsonr(cluster_values.flatten(), bright_values.flatten())
        print(f"Pearson correlation between cluster values and brighter zones: {corr}")

        return corr