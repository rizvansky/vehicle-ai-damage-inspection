import logging
import os
import time
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
from onnxconverter_common import float16
from optimum.onnxruntime import ORTModelForSemanticSegmentation

from src.models.components.semantic.deeplabv3plus import DeepLabV3PlusSemantic
from src.models.components.semantic.segformer import SegFormerSemantic
from src.models.components.semantic.unet import UNetSemantic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXExporter:
    """ONNX converter for semantic segmentation models."""

    def __init__(
        self, model_type: str, num_classes: int, input_shape: tuple[int, int, int, int]
    ) -> None:
        """Initialize ONNX converter.

        Args:
            model_type (str): Type of model (`"unet"`, `"deeplabv3plus"`, `"segformer"`).
            num_classes (int): Number of segmentation classes.
            input_shape (tuple[int, int, int, int]): Input tensor shape (batch_size, channels, height, width).
        """

        self.model_type = model_type.lower()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None

    def create_model(self, **kwargs) -> torch.nn.Module:
        """Create model instance based on model type.

        Args:
            **kwargs: Additional arguments for model initialization.

        Returns:
            torch.nn.Module: PyTorch model instance.

        Raises:
            ValueError: If model type is not supported.
        """

        if self.model_type == "unet":
            return UNetSemantic(
                num_classes=self.num_classes,
                encoder_name=kwargs.get("encoder_name", "resnet34"),
                encoder_weights=kwargs.get("encoder_weights", "imagenet"),
                in_channels=kwargs.get("in_channels", 3),
                activation=kwargs.get("activation", None),
                ignore_index=kwargs.get("ignore_index", 255),
            )
        elif self.model_type == "deeplabv3plus":
            return DeepLabV3PlusSemantic(
                num_classes=self.num_classes,
                encoder_name=kwargs.get("encoder_name", "resnet50"),
                encoder_weights=kwargs.get("encoder_weights", "imagenet"),
                in_channels=kwargs.get("in_channels", 3),
                activation=kwargs.get("activation", None),
                ignore_index=kwargs.get("ignore_index", 255),
            )
        elif self.model_type == "segformer":
            return SegFormerSemantic(
                num_classes=self.num_classes,
                backbone=kwargs.get("backbone", "nvidia/segformer-b0-finetuned-ade-512-512"),
                pretrained=kwargs.get("pretrained", True),
                ignore_index=kwargs.get("ignore_index", 255),
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def load_weights(self, model: torch.nn.Module, weights_path: str) -> torch.nn.Module:
        """Load weights from checkpoint file.

        Args:
            model (torch.nn.Module): PyTorch model instance.
            weights_path (str): Path to weights file.

        Returns:
            torch.nn.Module: Model with loaded weights.
        """

        logger.info(f"Loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
        logger.info("Weights loaded successfully")

        return model

    def convert_standard_onnx(
        self, model: torch.nn.Module, output_path: str, opset_version: int = 11
    ) -> None:
        """Convert model to ONNX using standard torch.onnx.export.

        Args:
            model (torch.nn.Module): PyTorch model to convert.
            output_path (str): Path to save ONNX model.
            opset_version (int, optional): ONNX opset version. Defaults to 11.
        """

        logger.info(f"Converting {self.model_type} to ONNX using standard export")

        dummy_input = torch.randn(self.input_shape)
        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={
                "pixel_values": {
                    0: "batch_size",
                    2: "height",
                    3: "width",
                },
                "logits": {
                    0: "batch_size",
                    2: "height",
                    3: "width",
                },
            },
        )
        logger.info(f"Model exported to {output_path}")

    def optimize_onnx_model(self, model_path: str) -> str:
        """Optimize ONNX model.

        Args:
            model_path (str): Path to ONNX model file.

        Returns:
            str: Path to optimized model (same as input if optimization fails).
        """

        logger.info("Optimizing ONNX model using 'onnxsim'")
        try:
            model = onnx.load(model_path)

            # Prepare input shapes for dynamic models
            input_shapes = {}
            for input_tensor in model.graph.input:
                input_name = input_tensor.name
                input_shape = []
                for dim in input_tensor.type.tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        input_shape.append(dim.dim_value)
                    elif dim.HasField("dim_param"):
                        if len(input_shape) == 0:  # batch dimension
                            input_shape.append(self.input_shape[0])
                        elif len(input_shape) == 1:  # channel dimension
                            input_shape.append(self.input_shape[1])
                        elif len(input_shape) == 2:  # height dimension
                            input_shape.append(self.input_shape[2])
                        elif len(input_shape) == 3:  # width dimension
                            input_shape.append(self.input_shape[3])
                        else:
                            input_shape.append(1)  # fallback
                    else:
                        input_shape.append(1)  # fallback for unknown dimensions
                input_shapes[input_name] = input_shape
            logger.info(f"Using input shapes for optimization: {input_shapes}")
            model_optimized, check = onnxsim.simplify(
                model,
                check_n=3,  # check the simplified model with 3 random inputs
                perform_optimization=True,
                skip_fuse_bn=False,
                skip_shape_inference=False,
                input_shapes=None,  # use dynamic shapes
                test_input_shapes=input_shapes,
            )
            if check:
                onnx.save(model_optimized, model_path)
                logger.info("ONNX model optimization completed successfully")
            else:
                logger.warning("onnxsim optimization check failed, using original model")
        except Exception as e:
            logger.warning(f"onnxsim optimization failed, using original model: {e}")

        return model_path

    def convert_to_fp16(self, model_path: str) -> str:
        """Convert ONNX model to FP16 precision.

        Args:
            model_path (str): Path to ONNX model file.

        Returns:
            str: Path to FP16 model (same as input if conversion fails).
        """

        logger.info("Converting ONNX model to FP16")
        try:
            model = onnx.load(model_path)
            model_fp16 = float16.convert_float_to_float16(
                model,
                min_positive_val=1e-7,
                max_finite_val=1e4,
                keep_io_types=True,  # keep input/output as FP32 for compatibility
                disable_shape_infer=False,
                op_block_list=None,
                node_block_list=None,
            )
            onnx.save(model_fp16, model_path)
            logger.info("ONNX model converted to FP16 successfully")
        except Exception as e:
            logger.warning(f"FP16 conversion failed, using original model: {e}")

        return model_path

    def convert_optimum_onnx(
        self,
        model_name_or_path: str,
        output_path: str,
    ) -> None:
        """Convert SegFormer model to ONNX using Optimum.

        Args:
            model_name_or_path (str): HuggingFace model name or local path.
            output_path (str): Directory to save ONNX model.
        """

        logger.info("Converting SegFormer to ONNX using Optimum")
        os.makedirs(output_path, exist_ok=True)
        ort_model = ORTModelForSemanticSegmentation.from_pretrained(
            model_name_or_path,
            export=True,
        )
        ort_model.save_pretrained(output_path)
        logger.info(f"Model exported to {output_path}")

    def validate_onnx_model(self, onnx_path: str) -> bool:
        """Validate ONNX model structure.

        Args:
            onnx_path (str): Path to ONNX model file.

        Returns:
            bool: `True` if model is valid, `False` otherwise.
        """

        try:
            logger.info(f"Validating ONNX model: {onnx_path}")
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            logger.info("ONNX model validation passed")
            return True
        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            return False

    def test_onnx_inference(
        self, onnx_path: str, original_model: torch.nn.Module | None = None
    ) -> dict[str, bool | float]:
        """Test ONNX model inference and compare with original model.

        Args:
            onnx_path (str): Path to ONNX model file.
            original_model (torch.nn.Module | None, optional): Original PyTorch model for comparison. Defaults to `None`.

        Returns:
            dict[str, bool | float]: Dictionary with test results.
        """

        results = {"inference_success": False, "max_diff": None}
        try:
            logger.info("Testing ONNX model inference")
            ort_session = ort.InferenceSession(onnx_path)
            test_input = np.random.randn(*self.input_shape).astype(np.float32)
            ort_inputs = {ort_session.get_inputs()[0].name: test_input}
            ort_outputs = ort_session.run(None, ort_inputs)
            onnx_output = ort_outputs[0]
            results["inference_success"] = True
            logger.info("ONNX inference successful")

            if original_model is not None:
                logger.info("Comparing ONNX output with original model")
                original_model.eval()
                with torch.no_grad():
                    torch_input = torch.from_numpy(test_input)
                    torch_output = original_model(torch_input)["logits"].numpy()

                    # Calculate maximum difference
                    max_diff = np.max(np.abs(onnx_output - torch_output))
                    results["max_diff"] = float(max_diff)
                    logger.info(f"Maximum difference between outputs: {max_diff}")
                    if max_diff < 1e-5:
                        logger.info("Outputs match within tolerance")
                    else:
                        logger.warning(f"Outputs differ by {max_diff}")

        except Exception as e:
            logger.error(f"ONNX inference test failed: {e}")
            results["inference_success"] = False

        return results

    def benchmark_model(self, model_path: str, num_runs: int = 50) -> dict[str, float]:
        """Benchmark ONNX model performance.

        Args:
            model_path (str): Path to ONNX model file.
            num_runs (int, optional): Number of benchmark runs. Defaults to 50.

        Returns:
            dict[str, float]: Dictionary with benchmark results.
        """

        logger.info(f"Benchmarking model: {model_path}")
        try:
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape

            # Handle dynamic shapes
            if any(isinstance(dim, str) for dim in input_shape):
                input_shape = list(self.input_shape)

            test_input = np.random.randn(*input_shape).astype(np.float32)

            # Warmup runs
            for _ in range(5):
                session.run(None, {input_name: test_input})

            # Benchmark runs
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                session.run(None, {input_name: test_input})
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms

            # Calculate statistics
            times = np.array(times)
            results = {
                "mean_ms": float(np.mean(times)),
                "std_ms": float(np.std(times)),
                "min_ms": float(np.min(times)),
                "max_ms": float(np.max(times)),
                "median_ms": float(np.median(times)),
                "throughput_fps": float(1000.0 / np.mean(times)),
            }

            logger.info(
                f"Benchmark completed - Mean: {results['mean_ms']:.2f}ms, FPS: {results['throughput_fps']:.2f}"
            )

            return results

        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {}

    def get_model_info(self, model_path: str) -> dict[str, int | float | str]:
        """Get detailed model information.

        Args:
            model_path (str): Path to ONNX model file.

        Returns:
            dict[str, int | float | str]: Dictionary with model information.
        """

        try:
            model = onnx.load(model_path)
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

            # Count parameters
            total_params = 0
            for initializer in model.graph.initializer:
                param_size = 1
                for dim in initializer.dims:
                    param_size *= dim
                total_params += param_size

            # Count nodes
            num_nodes = len(model.graph.node)

            # Get unique node types
            node_types = {node.op_type for node in model.graph.node}

            info = {
                "model_size_mb": model_size_mb,
                "total_parameters": total_params,
                "num_nodes": num_nodes,
                "node_types": list(node_types),
                "opset_version": model.opset_import[0].version if model.opset_import else "Unknown",
            }

            return info

        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}

    def convert_and_test(
        self,
        output_path: str,
        weights_path: str | None = None,
        enable_optimization: bool = True,
        enable_fp16: bool = False,
        enable_benchmarking: bool = False,
        benchmark_runs: int = 50,
        **model_kwargs,
    ) -> dict[str, bool | float]:
        """Convert model to ONNX and test the conversion.

        Args:
            output_path (str): Path to save ONNX model.
            weights_path (str | None): Optional path to model weights.
            enable_optimization (bool, optional): Whether to apply model optimization using onnxsim. Defaults to `True`.
            enable_fp16 (bool, optional): Whether to convert the model to FP16. Defaults to `False`.
            enable_benchmarking (bool, optional): Whether to benchmark the model. Defaults to `False`.
            benchmark_runs: Number of benchmark runs. Defaults to 50.
            **model_kwargs: Additional arguments for model creation.

        Returns:
            Dictionary with conversion and test results.
        """

        results = {
            "conversion_success": False,
            "validation_success": False,
            "inference_success": False,
            "optimization_success": False,
            "fp16_success": False,
            "max_diff": None,
            "benchmark_results": {},
        }
        try:
            logger.info(f"Creating {self.model_type} model")
            model = self.create_model(**model_kwargs)
            if weights_path:
                model = self.load_weights(model, weights_path)

            if self.model_type == "segformer":
                with TemporaryDirectory() as tmp_dir:
                    model.model.save_pretrained(tmp_dir)
                    self.convert_optimum_onnx(tmp_dir, output_path)
                    onnx_file = os.path.join(output_path, "model.onnx")
            else:
                self.convert_standard_onnx(model, output_path)
                onnx_file = output_path

            results["conversion_success"] = True
            logger.info("ONNX conversion completed successfully")

            if enable_optimization:
                logger.info("Applying onnxsim optimization")
                self.optimize_onnx_model(onnx_file)
                results["optimization_success"] = True

            if enable_fp16:
                logger.info("Converting to FP16")
                try:
                    self.convert_to_fp16(onnx_file)
                    results["fp16_success"] = True
                    logger.info("FP16 conversion completed successfully")
                except Exception as e:
                    logger.warning(f"FP16 conversion failed: {e}")
                    results["fp16_success"] = False

            if self.validate_onnx_model(onnx_file):
                results["validation_success"] = True
                test_results = self.test_onnx_inference(onnx_file, model)
                results.update(test_results)
                if enable_benchmarking:
                    logger.info("Running performance benchmarks")
                    benchmark_results = self.benchmark_model(onnx_file, benchmark_runs)
                    model_info = self.get_model_info(onnx_file)
                    results["benchmark_results"] = {**benchmark_results, **model_info}

        except Exception as e:
            logger.error(f"Conversion failed: {e}")

        return results
