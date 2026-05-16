import numpy as np
from typing import Tuple, Callable
from backward_propagation import DenseBackward, ReLUBackward, Conv2DBackward


class GradientChecker:
    
    EPSILON = 1e-5
    THRESHOLD = 1e-5
    
    @staticmethod
    def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_pred - y_true) ** 2)
    
    @staticmethod
    def mse_loss_gradient(y_true: np.ndarray, y_pred: np.ndarray, batch_size: int) -> np.ndarray:
        return (2 / batch_size) * (y_pred - y_true)
    
    @staticmethod
    def numerical_gradient(
        func: Callable,
        param: np.ndarray,
        epsilon: float = 1e-5
    ) -> np.ndarray:
        grad = np.zeros_like(param, dtype=float)
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:
            idx = it.multi_index
            old_value = param[idx]
            
            param[idx] = old_value + epsilon
            fxh_pos = func()
            
            param[idx] = old_value - epsilon
            fxh_neg = func()
            
            grad[idx] = (fxh_pos - fxh_neg) / (2 * epsilon)
            param[idx] = old_value
            
            it.iternext()
        
        return grad
    
    @staticmethod
    def relative_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
        diff = np.linalg.norm(analytical - numerical)
        denom = np.linalg.norm(analytical) + np.linalg.norm(numerical)
        return diff / (denom + 1e-8)
    
    @classmethod
    def check_dense_layer(
        cls,
        input_size: int = 10,
        output_size: int = 5,
        batch_size: int = 3
    ) -> dict:
        np.random.seed(42)
        
        x = np.random.randn(batch_size, input_size)
        weight = np.random.randn(input_size, output_size) * 0.01
        bias = np.random.randn(output_size) * 0.01
        y_true = np.eye(output_size)[np.random.randint(0, output_size, batch_size)]
        
        def forward():
            return np.dot(x, weight) + bias
        
        y_pred = forward()
        loss = cls.mse_loss(y_true, y_pred)
        dL_dy = cls.mse_loss_gradient(y_true, y_pred, batch_size)
        
        dL_dx_analytical, dL_dW_analytical, dL_db_analytical = DenseBackward.backward(
            dL_dy, x, weight
        )
        
        def func_weight():
            y = np.dot(x, weight) + bias
            return cls.mse_loss(y_true, y)
        
        dL_dW_numerical = cls.numerical_gradient(func_weight, weight, cls.EPSILON)
        
        error_weight = cls.relative_error(dL_dW_analytical, dL_dW_numerical)
        status_weight = "✓ PASS" if error_weight < cls.THRESHOLD else "✗ FAIL"
        
        return {
            "layer": "Dense",
            "input_shape": x.shape,
            "output_shape": y_pred.shape,
            "loss": float(loss),
            "weight_error": float(error_weight),
            "weight_status": status_weight,
            "threshold": cls.THRESHOLD,
            "passed": error_weight < cls.THRESHOLD
        }
    
    @classmethod
    def check_relu_layer(
        cls,
        input_shape: Tuple = (4, 8, 8, 16)
    ) -> dict:
        np.random.seed(42)
        
        x = np.random.randn(*input_shape) * 2 - 1
        dL_dy = np.random.randn(*input_shape)
        
        dL_dx_analytical = ReLUBackward.backward(dL_dy, x)
        
        def func():
            forward_out = np.maximum(0, x)
            return np.sum(forward_out * dL_dy)
        
        dL_dx_numerical = cls.numerical_gradient(func, x, cls.EPSILON)
        
        error = cls.relative_error(dL_dx_analytical, dL_dx_numerical)
        status = "✓ PASS" if error < cls.THRESHOLD else "✗ FAIL"
        
        return {
            "layer": "ReLU",
            "input_shape": x.shape,
            "error": float(error),
            "status": status,
            "threshold": cls.THRESHOLD,
            "passed": error < cls.THRESHOLD
        }
    
    @classmethod
    def check_conv2d_layer(
        cls,
        input_shape: Tuple = (2, 8, 8, 3),
        kernel_shape: Tuple = (3, 3, 3, 16),
        stride: int = 1,
        padding: int = 1
    ) -> dict:
        np.random.seed(42)
        
        x = np.random.randn(*input_shape) * 0.1
        kernel = np.random.randn(*kernel_shape) * 0.01
        dL_dy = np.random.randn(input_shape[0], 
                                (input_shape[1] + 2*padding - kernel_shape[0])//stride + 1,
                                (input_shape[2] + 2*padding - kernel_shape[1])//stride + 1,
                                kernel_shape[3]) * 0.1
        
        dL_dx_analytical, dL_dW_analytical, _ = Conv2DBackward.backward(
            dL_dy, x, kernel, stride, padding
        )
        
        def func_kernel():
            from forward_propagation import ForwardPropagation
            fp = ForwardPropagation()
            fp.add_conv2d(kernel, np.zeros(kernel_shape[3]), stride=stride, padding=padding)
            output = fp.forward(x)
            return np.sum(output * dL_dy)
        
        dL_dW_numerical = cls.numerical_gradient(func_kernel, kernel, cls.EPSILON * 10)
        
        error_kernel = cls.relative_error(dL_dW_analytical, dL_dW_numerical)
        status_kernel = "✓ PASS" if error_kernel < cls.THRESHOLD * 10 else "✗ FAIL"
        
        return {
            "layer": "Conv2D",
            "input_shape": x.shape,
            "kernel_shape": kernel.shape,
            "kernel_error": float(error_kernel),
            "kernel_status": status_kernel,
            "threshold": cls.THRESHOLD * 10,
            "passed": error_kernel < cls.THRESHOLD * 10
        }


def run_all_gradient_checks() -> dict:
    results = {
        "dense": GradientChecker.check_dense_layer(),
        "relu": GradientChecker.check_relu_layer(),
        "conv2d": GradientChecker.check_conv2d_layer()
    }
    
    all_passed = all(r.get("passed", False) for r in results.values())
    results["summary"] = {
        "all_passed": all_passed,
        "status": "✓ ALL TESTS PASSED" if all_passed else "✗ SOME TESTS FAILED"
    }
    
    return results
