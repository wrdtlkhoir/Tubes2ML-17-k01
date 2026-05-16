import numpy as np
from typing import Tuple


class ReLUBackward:
    @staticmethod
    def backward(dL_dy: np.ndarray, x: np.ndarray) -> np.ndarray:
        mask = (x > 0).astype(float)
        return dL_dy * mask


class DenseBackward:
    @staticmethod
    def backward(dL_dy: np.ndarray, x: np.ndarray, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dL_dx = np.dot(dL_dy, weight.T)
        dL_dW = np.dot(x.T, dL_dy)
        dL_db = np.mean(dL_dy, axis=0)
        return dL_dx, dL_dW, dL_db


class Conv2DBackward:
    @staticmethod
    def backward(
        dL_dy: np.ndarray,
        x: np.ndarray,
        kernel: np.ndarray,
        stride: int = 1,
        padding: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N, H, W, C = x.shape
        kH, kW, C_in, C_out = kernel.shape
        
        if padding > 0:
            x = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
            H = H + 2 * padding
            W = W + 2 * padding
        
        out_h = (H - kH) // stride + 1
        out_w = (W - kW) // stride + 1
        
        dL_dx = np.zeros_like(x)
        dL_dW = np.zeros_like(kernel)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                
                patch = x[:, h_start:h_start+kH, w_start:w_start+kW, :]
                patch_reshaped = patch.reshape(N, -1)
                
                grad_out = dL_dy[:, i, j, :]
                
                kernel_reshaped = kernel.reshape(-1, C_out)
                dL_dx[:, h_start:h_start+kH, w_start:w_start+kW, :] += np.dot(
                    grad_out, kernel_reshaped.T
                ).reshape(N, kH, kW, C_in)
                
                dL_dW += np.dot(patch_reshaped.T, grad_out).reshape(kH, kW, C_in, C_out)
        
        dL_db = np.sum(dL_dy, axis=(0, 1, 2))
        
        return dL_dx, dL_dW, dL_db


class MaxPoolingBackward:
    @staticmethod
    def backward(
        dL_dy: np.ndarray,
        x: np.ndarray,
        pool_h: int = 2,
        pool_w: int = 2,
        stride: Tuple[int, int] = None
    ) -> np.ndarray:
        if stride is None:
            stride = (pool_h, pool_w)
        
        stride_h, stride_w = stride
        N, H, W, C = x.shape
        out_h = (H - pool_h) // stride_h + 1
        out_w = (W - pool_w) // stride_w + 1
        
        dL_dx = np.zeros_like(x)
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride_h
                w_start = j * stride_w
                h_end = h_start + pool_h
                w_end = w_start + pool_w
                
                pool_region = x[:, h_start:h_end, w_start:w_end, :]
                max_indices = np.argmax(pool_region.reshape(N, -1, C), axis=1)
                
                for n in range(N):
                    for c in range(C):
                        max_idx = max_indices[n, c]
                        h_offset = max_idx // pool_w
                        w_offset = max_idx % pool_w
                        
                        dL_dx[n, h_start + h_offset, w_start + w_offset, c] = dL_dy[n, i, j, c]
        
        return dL_dx


class AveragePoolingBackward:
    @staticmethod
    def backward(
        dL_dy: np.ndarray,
        x: np.ndarray,
        pool_h: int = 2,
        pool_w: int = 2,
        stride: Tuple[int, int] = None
    ) -> np.ndarray:
        if stride is None:
            stride = (pool_h, pool_w)
        
        stride_h, stride_w = stride
        N, H, W, C = x.shape
        pool_size = pool_h * pool_w
        
        dL_dx = np.zeros_like(x)
        
        for i in range((H - pool_h) // stride_h + 1):
            for j in range((W - pool_w) // stride_w + 1):
                h_start = i * stride_h
                w_start = j * stride_w
                h_end = h_start + pool_h
                w_end = w_start + pool_w
                
                dL_dx[:, h_start:h_end, w_start:w_end, :] += dL_dy[:, i:i+1, j:j+1, :] / pool_size
        
        return dL_dx


class GlobalAvgPoolingBackward:
    @staticmethod
    def backward(dL_dy: np.ndarray, x: np.ndarray) -> np.ndarray:
        N, H, W, C = x.shape
        pool_size = H * W
        
        dL_dx = np.zeros_like(x)
        for n in range(N):
            for c in range(C):
                dL_dx[n, :, :, c] = dL_dy[n, c] / pool_size
        
        return dL_dx
