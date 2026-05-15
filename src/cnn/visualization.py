import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Optional


class FeatureMapVisualizer:
    
    @staticmethod
    def visualize_feature_maps(feature_maps: np.ndarray, 
                               layer_name: str = "Conv Layer",
                               max_filters: int = 16) -> None:
        num_channels = feature_maps.shape[2]
        num_to_show = min(max_filters, num_channels)
        
        # Hitung grid dimensi
        grid_size = int(np.ceil(np.sqrt(num_to_show)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(num_to_show):
            feature_map = feature_maps[:, :, i]
            
            # Normalize untuk visualization
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f'Filter {i}')
            axes[i].axis('off')
        
        # Hapus subplot kosong
        for i in range(num_to_show, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'{layer_name} - Feature Maps (showing {num_to_show}/{num_channels})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_multiple_layers(feature_maps_list: List[np.ndarray],
                                  layer_names: List[str],
                                  filters_per_layer: int = 8) -> None:
        num_layers = len(feature_maps_list)
        
        fig, axes = plt.subplots(num_layers, filters_per_layer, 
                                 figsize=(filters_per_layer*2, num_layers*2))
        
        if num_layers == 1:
            axes = axes.reshape(1, -1)
        
        for layer_idx, (feature_maps, layer_name) in enumerate(zip(feature_maps_list, layer_names)):
            num_filters = feature_maps.shape[2]
            
            for filter_idx in range(filters_per_layer):
                if filter_idx < num_filters:
                    feature_map = feature_maps[:, :, filter_idx]
                    # Normalize
                    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                    
                    axes[layer_idx, filter_idx].imshow(feature_map, cmap='viridis')
                    axes[layer_idx, filter_idx].set_title(f'{layer_name} F{filter_idx}', fontsize=8)
                else:
                    axes[layer_idx, filter_idx].set_title(f'{layer_name} (N/A)', fontsize=8)
                
                axes[layer_idx, filter_idx].axis('off')
        
        plt.tight_layout()
        plt.show()


class GradCAM:
    def __init__(self, feature_extractor_fn, classifier_fn):
        self.feature_extractor_fn = feature_extractor_fn
        self.classifier_fn = classifier_fn
    
    def compute_gradcam(self, image: np.ndarray, target_class: int) -> np.ndarray:
        image_batch = image[np.newaxis, :, :, :]
        feature_maps = self.feature_extractor_fn(image_batch)  # (1, h, w, c)
        feature_maps = feature_maps[0]  # (h, w, c)
        
        num_channels = feature_maps.shape[2]
        h, w, c = feature_maps.shape
        
        weights = np.zeros(num_channels)
        
        for channel_idx in range(num_channels):
            eps = 1e-4
        
            feature_maps_plus = feature_maps.copy()
            score_plus = self._get_class_score(feature_maps_plus, target_class)
            
            feature_maps_minus = feature_maps.copy()
            feature_maps_minus[:, :, channel_idx] -= eps
            score_minus = self._get_class_score(feature_maps_minus, target_class)
            
            gradient = (score_plus - score_minus) / eps
            
            weights[channel_idx] = np.mean(gradient)
        
        cam = np.zeros((h, w))
        for channel_idx in range(num_channels):
            cam += weights[channel_idx] * feature_maps[:, :, channel_idx]
        
        cam = np.maximum(cam, 0)
        
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def _get_class_score(self, feature_maps: np.ndarray, target_class: int) -> float:
        
        flattened = feature_maps.flatten()
        
        scores = self.classifier_fn(flattened)
        
        return scores[target_class]
    
    @staticmethod
    def overlay_cam_on_image(image: np.ndarray, cam: np.ndarray, 
                            alpha: float = 0.5) -> np.ndarray:
        from scipy.ndimage import zoom
        
        if image.ndim == 3:
            h_img, w_img = image.shape[:2]
        else:
            h_img, w_img = image.shape
            image = np.stack([image]*3, axis=-1)
        
        h_cam, w_cam = cam.shape
        
        if (h_cam, w_cam) != (h_img, w_img):
            scale_h = h_img / h_cam
            scale_w = w_img / w_cam
            cam_resized = zoom(cam, (scale_h, scale_w), order=1)
        else:
            cam_resized = cam
        
        if image.max() > 1:
            image_norm = image / 255.0
        else:
            image_norm = image
      
        heatmap = cm.jet(cam_resized)[:, :, :3]
        
        overlay = alpha * heatmap + (1 - alpha) * image_norm
        overlay = np.clip(overlay, 0, 1)
        
        return overlay
    
    @staticmethod
    def visualize_gradcam(image: np.ndarray, cam: np.ndarray,
                         title: str = "Grad-CAM Visualization") -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        if image.ndim == 3 and image.shape[2] == 3:
            if image.max() > 1:
                axes[0].imshow(image.astype(np.uint8))
            else:
                axes[0].imshow(image)
        else:
            axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Grad-CAM heatmap
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        cbar1 = plt.colorbar(axes[1].images[0], ax=axes[1])
        
        # Overlay
        overlay = GradCAM.overlay_cam_on_image(image, cam, alpha=0.5)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


class ActivationHeatmap:
    @staticmethod
    def compute_activation_heatmap(feature_maps: np.ndarray) -> np.ndarray:
        heatmap = np.mean(feature_maps, axis=2)
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap
    
    @staticmethod
    def visualize_activation_progression(activations_dict: dict,
                                        image: np.ndarray = None) -> None:
        num_layers = len(activations_dict)
        
        if image is not None:
            fig, axes = plt.subplots(1, num_layers + 1, figsize=(3*(num_layers+1), 3))
            
            # Original image
            if image.ndim == 3 and image.shape[2] == 3:
                if image.max() > 1:
                    axes[0].imshow(image.astype(np.uint8))
                else:
                    axes[0].imshow(image)
            else:
                axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            start_idx = 1
        else:
            fig, axes = plt.subplots(1, num_layers, figsize=(3*num_layers, 3))
            if num_layers == 1:
                axes = np.array([axes])
            start_idx = 0
        
        for layer_idx, (layer_name, feature_maps) in enumerate(activations_dict.items()):
            heatmap = ActivationHeatmap.compute_activation_heatmap(feature_maps)
            
            ax = axes[start_idx + layer_idx]
            im = ax.imshow(heatmap, cmap='hot')
            ax.set_title(f'{layer_name}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        
        plt.suptitle('Activation Progression Through Layers', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
