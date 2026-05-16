import numpy as np
import time
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass


@dataclass
class InferenceResult:
    predictions: np.ndarray
    execution_time: float
    throughput: float
    average_latency: float


class BatchInference:
    
    @staticmethod
    def forward_batch(
        model: Callable,
        x: np.ndarray,
        batch_size: int = 32
    ) -> InferenceResult:
        num_samples = x.shape[0]
        predictions = []
        
        start_time = time.time()
        
        for i in range(0, num_samples, batch_size):
            batch = x[i:i+batch_size]
            batch_pred = model(batch)
            predictions.append(batch_pred)
        
        execution_time = time.time() - start_time
        
        predictions = np.concatenate(predictions, axis=0)
        
        throughput = num_samples / execution_time
        avg_latency = (execution_time / num_samples) * 1000
        
        return InferenceResult(
            predictions=predictions,
            execution_time=execution_time,
            throughput=throughput,
            average_latency=avg_latency
        )
    
    @staticmethod
    def compare_models(
        model1: Callable,
        model2: Callable,
        x: np.ndarray,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2",
        batch_size: int = 32,
        num_runs: int = 1
    ) -> Dict:
        results = {
            "model1_name": model1_name,
            "model2_name": model2_name,
            "num_samples": x.shape[0],
            "batch_size": batch_size,
            "num_runs": num_runs
        }
        
        times1 = []
        times2 = []
        
        for run in range(num_runs):
            result1 = BatchInference.forward_batch(model1, x, batch_size)
            result2 = BatchInference.forward_batch(model2, x, batch_size)
            
            times1.append(result1.execution_time)
            times2.append(result2.execution_time)
            
            if run == 0:
                results["predictions_match"] = np.allclose(
                    result1.predictions,
                    result2.predictions,
                    atol=1e-5
                )
                results["max_diff"] = np.max(np.abs(result1.predictions - result2.predictions))
        
        avg_time1 = np.mean(times1)
        avg_time2 = np.mean(times2)
        std_time1 = np.std(times1)
        std_time2 = np.std(times2)
        
        results["model1"] = {
            "avg_time": avg_time1,
            "std_time": std_time1,
            "throughput": x.shape[0] / avg_time1,
            "avg_latency_ms": (avg_time1 / x.shape[0]) * 1000
        }
        
        results["model2"] = {
            "avg_time": avg_time2,
            "std_time": std_time2,
            "throughput": x.shape[0] / avg_time2,
            "avg_latency_ms": (avg_time2 / x.shape[0]) * 1000
        }
        
        speedup = avg_time1 / avg_time2
        faster_model = model1_name if speedup > 1 else model2_name
        
        results["speedup"] = {
            "ratio": speedup,
            "faster_model": faster_model,
            "percentage": (abs(speedup - 1) * 100) if speedup > 1 else ((1 / speedup - 1) * 100)
        }
        
        return results
    
    @staticmethod
    def timing_analysis(
        model: Callable,
        x_list: List[np.ndarray],
        size_labels: List[str],
        batch_size: int = 32,
        num_runs: int = 3
    ) -> Dict:
        results = {
            "batch_size": batch_size,
            "num_runs": num_runs,
            "sizes": []
        }
        
        for x, label in zip(x_list, size_labels):
            times = []
            
            for _ in range(num_runs):
                result = BatchInference.forward_batch(model, x, batch_size)
                times.append(result.execution_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            results["sizes"].append({
                "label": label,
                "num_samples": x.shape[0],
                "avg_time": avg_time,
                "std_time": std_time,
                "throughput": x.shape[0] / avg_time,
                "avg_latency_ms": (avg_time / x.shape[0]) * 1000
            })
        
        return results
    
    @staticmethod
    def confidence_analysis(
        predictions: np.ndarray,
        labels: np.ndarray = None
    ) -> Dict:
        if len(predictions.shape) == 1:
            confidence = predictions
        else:
            confidence = np.max(predictions, axis=1)
        
        results = {
            "mean_confidence": float(np.mean(confidence)),
            "std_confidence": float(np.std(confidence)),
            "min_confidence": float(np.min(confidence)),
            "max_confidence": float(np.max(confidence)),
            "median_confidence": float(np.median(confidence))
        }
        
        if labels is not None:
            predicted_labels = np.argmax(predictions, axis=1)
            correct = predicted_labels == labels
            
            correct_confidence = confidence[correct]
            incorrect_confidence = confidence[~correct]
            
            if len(correct_confidence) > 0:
                results["correct_mean_confidence"] = float(np.mean(correct_confidence))
            if len(incorrect_confidence) > 0:
                results["incorrect_mean_confidence"] = float(np.mean(incorrect_confidence))
            
            results["accuracy"] = float(np.mean(correct))
            results["num_correct"] = int(np.sum(correct))
            results["num_total"] = len(labels)
        
        return results
    
    @staticmethod
    def memory_analysis(
        model: Callable,
        input_shape: Tuple,
        batch_sizes: List[int] = None
    ) -> Dict:
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16, 32]
        
        results = {
            "input_shape": input_shape,
            "batch_analyses": []
        }
        
        for batch_size in batch_sizes:
            x = np.random.randn(batch_size, *input_shape)
            
            input_memory = x.nbytes / (1024 ** 2)
            
            output = model(x)
            output_memory = output.nbytes / (1024 ** 2)
            
            results["batch_analyses"].append({
                "batch_size": batch_size,
                "input_memory_mb": input_memory,
                "output_memory_mb": output_memory,
                "total_memory_mb": input_memory + output_memory
            })
        
        return results


class ModelEvaluator:
    
    @staticmethod
    def evaluate_keras_vs_scratch(
        keras_model,
        scratch_model: Callable,
        x_test: np.ndarray,
        y_test: np.ndarray = None,
        batch_size: int = 32
    ) -> Dict:
        def keras_forward(x):
            return keras_model.predict(x, verbose=0)
        
        comparison_results = BatchInference.compare_models(
            keras_forward,
            scratch_model,
            x_test,
            model1_name="Keras",
            model2_name="From-Scratch",
            batch_size=batch_size
        )
        
        results = {
            "comparison": comparison_results
        }
        
        if y_test is not None:
            keras_pred = keras_forward(x_test)
            scratch_pred = scratch_model(x_test)
            
            results["keras_confidence"] = BatchInference.confidence_analysis(
                keras_pred, y_test
            )
            results["scratch_confidence"] = BatchInference.confidence_analysis(
                scratch_pred, y_test
            )
        
        return results
