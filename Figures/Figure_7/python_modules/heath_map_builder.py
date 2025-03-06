from ultralytics import YOLO
import torch
import cv2
import numpy as np


class HeatMapBuilder:
    
    def __init__(self, model_path:str):
        
        self.model_path = model_path
        
        try:
            self.model = YOLO(model_path) 
            print("The Yolo model was successfully loaded")
        except Exception as e:
            print(f"Error when initializing the HetMapBuilder: {e}")
            
        self.activations = {}
        self.layer_names = None
        self.layer = None
        self.input_tensor = None
        self.raw_heat_map = None
        
    # Hook to capture activations
    def forward_hook(self, name):
        def hook(module, input, output):
            global activations
            self.activations[name] = output  # Store activations by layer name
        return hook

    def get_layer_names(self):
        self.layer_names = [name for name, module in self.model.model.named_modules() if hasattr(module, "forward")]

    def get_register_hooks_for_layers(self):
        # Register hooks for all layers
        for layer_name in self.layer_names:
            self.layer = dict(self.model.model.named_modules())[layer_name]
            self.layer.register_forward_hook(self.forward_hook(layer_name))

    def load_image(self, image_path: str):
        try:
            # Load image
            self.image = cv2.imread(image_path)
            if self.image is None:
                raise FileNotFoundError(f"Image not found at {image_path}")
            
            # Convert BGR to RGB
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            
            # Normalize and convert to tensor
            self.input_tensor = torch.from_numpy(self.image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            print("Image successfully loaded")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise


    def get_heat_map(self, heath_map_file_name="./images/heatmap_combined.jpg", save_all_heathmaps=False, overlay_image=False):
        
        # mandatory steps
        self.get_layer_names()
        self.get_register_hooks_for_layers()
        
        # Pass the image through the model
        try:
            self.model(self.input_tensor)
        except Exception as e:
            print(f"An exception occurred: {e} trying to convert to a different input tensor shape")
            self.model(self.input_tensor)
            
        # Initialize combined activation map
        combined_activation_map = None
        heatmaps = {}  # Store individual heatmaps

        # Process activations for each layer
        for layer_name, activation in self.activations.items():
            activation_map = activation[0].detach().cpu().numpy()  # Select the first batch
            activation_map = np.mean(activation_map, axis=0)  # Average across channels

            # Normalize the activation map
            activation_map -= activation_map.min()
            activation_map /= activation_map.max()

            # Resize to match the original image size
            activation_map_resized = cv2.resize(activation_map, (self.image.shape[1], self.image.shape[0]))

            # Save individual heatmaps
            heatmaps[layer_name] = activation_map_resized

            # Combine activation maps
            if combined_activation_map is None:
                combined_activation_map = activation_map_resized
            else:
                combined_activation_map += activation_map_resized

        # Final normalization of the combined activation map
        if combined_activation_map is not None:
            combined_activation_map -= combined_activation_map.min()
            combined_activation_map /= combined_activation_map.max()

        if save_all_heathmaps:
            # Save individual heatmaps
            for layer_name, heatmap in heatmaps.items():
                heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(f"heatmap_{layer_name}.jpg", heatmap_color)
                print(f"Saved heatmap for {layer_name} at heatmap_{layer_name}.jpg")

        # Save combined heatmap if applicable
        if combined_activation_map is not None:
            self.raw_heat_map = combined_activation_map
            print("Generating the heath map ... ")
            combined_heatmap_color = cv2.applyColorMap((combined_activation_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(heath_map_file_name, combined_heatmap_color)
            print("Saved combined heatmap at heatmap_combined.jpg")
            
            if overlay_image:
                overlay_combined = cv2.addWeighted(self.image, 0.5, combined_heatmap_color, 0.5, 0)
                cv2.imwrite("heatmap_combined_overlay.jpg", overlay_combined)