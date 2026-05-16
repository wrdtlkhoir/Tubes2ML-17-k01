import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import zoom
from typing import List, Tuple, Optional, Callable


class FeatureMapVisualizer:
    
    @staticmethod
    def visualize_feature_maps(
        feature_maps: np.ndarray,
        layer_name: str = "Conv Layer",
        max_filters: int = 16,
        figsize: Tuple = (16, 8)
    ) -> None:
        feature_maps = feature_maps.squeeze()
        num_filters = min(feature_maps.shape[2], max_filters)
        
        fig, axes = plt.subplots(2, max_filters//2, figsize=figsize)
        axes = axes.flatten()
        
        for i in range(num_filters):
            ax = axes[i]
            feature = feature_maps[:, :, i]
            
            im = ax.imshow(feature, cmap='viridis')
            ax.set_title(f'Filter {i+1}', fontsize=10, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'{layer_name} - Feature Maps', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_multiple_layers(
        feature_maps_list: List[np.ndarray],
        layer_names: List[str],
        filters_per_layer: int = 4,
        figsize: Tuple = (16, 6)
    ) -> None:
        num_layers = len(feature_maps_list)
        fig, axes = plt.subplots(num_layers, filters_per_layer, figsize=figsize)
        
        if num_layers == 1:
            axes = axes.reshape(1, -1)
        
        for layer_idx, (features, name) in enumerate(zip(feature_maps_list, layer_names)):
            features = features.squeeze()
            
            for filter_idx in range(filters_per_layer):
                ax = axes[layer_idx, filter_idx]
                feature = features[:, :, filter_idx]
                
                im = ax.imshow(feature, cmap='viridis')
                ax.set_title(f'{name} F{filter_idx+1}', fontsize=9)
                ax.axis('off')
        
        plt.suptitle('Feature Map Progression Across Layers', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


class GradCAM:
    
    def __init__(
        self,
        feature_extractor: Callable,
        classifier: Callable
    ):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
    
    def compute_gradcam(
        self,
        image: np.ndarray,
        target_class: int,
        epsilon: float = 1e-6
    ) -> np.ndarray:
        features_batch = self.feature_extractor(image)
        
        features = np.squeeze(features_batch)
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=-1)
        
        scores = self.classifier(features_batch.flatten())
        target_score = scores[target_class]
        
        num_channels = features.shape[-1]
        weights = np.zeros(num_channels)
        
        for c in range(num_channels):
            feature_c = features[:, :, c]
            activation_sum = np.sum(feature_c)
            weights[c] = activation_sum if activation_sum > epsilon else epsilon
        
        weights = weights / (np.sum(weights) + epsilon)
        
        cam = np.zeros((features.shape[0], features.shape[1]))
        for c in range(num_channels):
            cam += weights[c] * features[:, :, c]
        
        cam = np.maximum(cam, 0)
        cam_min = np.min(cam)
        cam_max = np.max(cam)
        
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        
        return cam
    
    @staticmethod
    def overlay_cam_on_image(
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: str = 'jet'
    ) -> np.ndarray:
        cmap = cm.get_cmap(colormap)
        
        if image.shape[2] == 3:
            if np.max(image) > 1:
                image = image / 255.0
        
        if cam.shape != image.shape[:2]:
            scale_h = image.shape[0] / cam.shape[0]
            scale_w = image.shape[1] / cam.shape[1]
            cam = zoom(cam, (scale_h, scale_w), order=1)
        
        heatmap = cmap(cam)[:, :, :3]
        overlay = (1 - alpha) * image[:, :, :3] + alpha * heatmap
        
        return np.clip(overlay, 0, 1)
    
    @staticmethod
    def visualize_gradcam(
        image: np.ndarray,
        cam: np.ndarray,
        title: str = "Grad-CAM",
        figsize: Tuple = (12, 4)
    ) -> None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        im = axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        overlay = GradCAM.overlay_cam_on_image(image, cam, alpha=0.5)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


class ActivationHeatmap:
    
    @staticmethod
    def compute_activation_heatmap(
        feature_maps: np.ndarray,
        method: str = 'mean'
    ) -> np.ndarray:
        if method == 'mean':
            heatmap = np.mean(np.abs(feature_maps), axis=2)
        elif method == 'max':
            heatmap = np.max(np.abs(feature_maps), axis=2)
        elif method == 'l2':
            heatmap = np.linalg.norm(feature_maps, axis=2)
        else:
            heatmap = np.mean(feature_maps, axis=2)
        
        heatmap_min = np.min(heatmap)
        heatmap_max = np.max(heatmap)
        
        if heatmap_max > heatmap_min:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        
        return heatmap
    
    @staticmethod
    def visualize_heatmap(
        heatmap: np.ndarray,
        title: str = "Activation Heatmap",
        figsize: Tuple = (8, 6)
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(heatmap, cmap='hot')
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Activation')
        plt.tight_layout()
        plt.show()


class LayerActivationTracker:
    
    def __init__(self):
        self.activations = {}
    
    def register_activation(self, layer_name: str, activation: np.ndarray):
        self.activations[layer_name] = activation
    
    def get_activation(self, layer_name: str) -> np.ndarray:
        return self.activations.get(layer_name)
    
    def clear(self):
        self.activations.clear()
    
    def visualize_progression(
        self,
        layer_names: Optional[List[str]] = None,
        figsize: Tuple = (14, 8)
    ) -> None:
        if layer_names is None:
            layer_names = list(self.activations.keys())
        
        num_layers = len(layer_names)
        fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, layer_name in enumerate(layer_names):
            ax = axes[idx]
            activation = self.get_activation(layer_name)
            
            if activation is not None:
                if len(activation.shape) == 3:
                    heatmap = ActivationHeatmap.compute_activation_heatmap(activation)
                    im = ax.imshow(heatmap, cmap='hot')
                elif len(activation.shape) == 2:
                    im = ax.imshow(activation, cmap='hot')
                else:
                    ax.text(0.5, 0.5, 'Unsupported shape', ha='center', va='center')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    continue
                
                ax.set_title(f'{layer_name}', fontsize=11, fontweight='bold')
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        for idx in range(len(layer_names), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Activation Progression Through Network', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
