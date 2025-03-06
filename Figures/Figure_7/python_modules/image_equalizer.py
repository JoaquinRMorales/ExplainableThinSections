from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import cv2
from scipy.stats import pearsonr
from sklearn.cluster import DBSCAN

class ImageEqualizer:
    
    def __init__(self, image_path: str, raw_heat_map: np.array):
        
        self.heatmap = raw_heat_map
        self.image_path = image_path
        self.gray_image = None
        self.equalized_image = None
        self.binary_image = None
        self.contours = None
        self.contour_image = None
        
    def load_image(self):
        # Read and preprocess the input image
        try:
            self.image = Image.open(self.image_path)
        except Exception as e:
            print(f"Error when loading the image {self.image_path}: {e}")
            
    def image_to_binary(self, threshold_value=0.5):
        
        # load image
        self.load_image()
        
        # Convert to grayscale
        self.gray_image = self.image.convert("L")

        # Apply histogram equalization to exacerbate brighter zones
        self.equalized_image = exposure.equalize_hist(np.array(self.gray_image))

        # Threshold to "black out" lower value zones
        # Adjust as needed
        self.binary_image = (self.equalized_image > threshold_value).astype(float)

    def get_contours(self):
        # Load the image
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        # Threshold the image to isolate white objects
        _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

        # Detect contours
        self.contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert original image to color for visualization
        self.contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def plot_contours(self, save_path=None):
        # Draw contours on the image
        cv2.drawContours(self.contour_image, self.contours, -1, (0, 255, 0), 2)

        # Display the image with contours
        plt.imshow(cv2.cvtColor(self.contour_image, cv2.COLOR_BGR2RGB))
        plt.title("Contours Detected")
        plt.axis("off")
        if save_path:
            # Save the figure to the specified path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print("Image succesfully save")
        plt.show()
    
    # def get_dbscan(self, threshold = 90):
    #     self.get_values_by_threshold(threshold)
    #     # Perform DBSCAN clustering on these points
    #     dbscan_auto = DBSCAN(eps=20, min_samples=10).fit(self.high_values_auto)
    #     self.labels_auto = dbscan_auto.labels_
        
    def plot_images(self, save_path=None):
    
        # Display the original and processed images
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(self.image)
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(self.equalized_image, cmap="gray")
        ax[1].set_title("Equalized Image")
        ax[1].axis("off")

        ax[2].imshow(self.binary_image, cmap="gray")
        ax[2].set_title("Binary Image")
        ax[2].axis("off")

        plt.tight_layout()
        if save_path:
            # Save the figure to the specified path
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print("Image succesfully save")
        plt.show()
        
        