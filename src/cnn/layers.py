import numpy as np
from typing import Tuple, Optional


class Conv2D:
    def __init__(self, kernel: np.ndarray, bias: np.ndarray, stride: int = 1, padding: int = 0):
        self.kernel = kernel
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.kH, self.kW, self.C_in, self.C_out = kernel.shape
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        N, H, W, C = x.shape
        
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (self.padding, self.padding), 
                          (self.padding, self.padding), (0, 0)), mode='constant')
            H = H + 2 * self.padding
            W = W + 2 * self.padding
        
        out_h = (H - self.kH) // self.stride + 1
        out_w = (W - self.kW) // self.stride + 1
        
        output = np.zeros((N, out_h, out_w, self.C_out))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                
                patch = x[:, h_start:h_start+self.kH, w_start:w_start+self.kW, :]
                patch = patch.reshape(N, -1)
                kernel_reshaped = self.kernel.reshape(-1, self.C_out)
                
                output[:, i, j, :] = np.dot(patch, kernel_reshaped) + self.bias
        
        return output


class LocallyConnected2D:
    def __init__(self, kernel: np.ndarray, bias: np.ndarray, stride: int = 1, padding: int = 0):
        self.kernel = kernel
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.out_rows, self.out_cols, self.kH_kW_C_in, self.C_out = kernel.shape
        
        self.kH = int(np.sqrt(self.kH_kW_C_in / self.kernel.shape[2]))
        self.kW = self.kH
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        N, H, W, C = x.shape
        
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (self.padding, self.padding), 
                          (self.padding, self.padding), (0, 0)), mode='constant')
            H = H + 2 * self.padding
            W = W + 2 * self.padding
        
        out_h = (H - self.kH) // self.stride + 1
        out_w = (W - self.kW) // self.stride + 1
        
        output = np.zeros((N, out_h, out_w, self.C_out))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                
                patch = x[:, h_start:h_start+self.kH, w_start:w_start+self.kW, :]
                patch = patch.reshape(N, -1)
                
                pos_idx = i * out_w + j
                kernel_at_pos = self.kernel[pos_idx, :, :]
                
                output[:, i, j, :] = np.dot(patch, kernel_at_pos) + self.bias[pos_idx, :]
        
        return output


class MaxPooling2D:
    def __init__(self, pool_size: Tuple[int, int] = (2, 2), strides: Tuple[int, int] = None):
        self.pool_h, self.pool_w = pool_size
        self.stride_h, self.stride_w = strides if strides else pool_size
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        N, H, W, C = x.shape
        
        out_h = (H - self.pool_h) // self.stride_h + 1
        out_w = (W - self.pool_w) // self.stride_w + 1
        
        output = np.zeros((N, out_h, out_w, C))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride_h
                w_start = j * self.stride_w
                
                pool_region = x[:, h_start:h_start+self.pool_h, w_start:w_start+self.pool_w, :]
                output[:, i, j, :] = np.max(pool_region, axis=(1, 2))
        
        return output


class AveragePooling2D:
    def __init__(self, pool_size: Tuple[int, int] = (2, 2), strides: Tuple[int, int] = None):
        self.pool_h, self.pool_w = pool_size
        self.stride_h, self.stride_w = strides if strides else pool_size
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        N, H, W, C = x.shape
        
        out_h = (H - self.pool_h) // self.stride_h + 1
        out_w = (W - self.pool_w) // self.stride_w + 1
        
        output = np.zeros((N, out_h, out_w, C))
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride_h
                w_start = j * self.stride_w
                
                pool_region = x[:, h_start:h_start+self.pool_h, w_start:w_start+self.pool_w, :]
                output[:, i, j, :] = np.mean(pool_region, axis=(1, 2))
        
        return output


class GlobalAveragePooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        N, H, W, C = x.shape
        return np.mean(x, axis=(1, 2))


class GlobalMaxPooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        N, H, W, C = x.shape
        return np.max(x, axis=(1, 2))


class Flatten:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(x.shape[0], -1)


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class Softmax:
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class Dense:
    def __init__(self, weight: np.ndarray, bias: np.ndarray, activation: Optional[str] = None):
        self.weight = weight
        self.bias = bias
        self.activation = activation
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        output = np.dot(x, self.weight) + self.bias
        
        if self.activation == 'relu':
            output = ReLU().forward(output)
        elif self.activation == 'softmax':
            output = Softmax().forward(output)
        
        return output
