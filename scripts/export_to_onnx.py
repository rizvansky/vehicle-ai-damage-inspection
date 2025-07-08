import argparse
import logging
import sys

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.export.onnx_exporter import ONNXExporter  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Export semantic segmentation models to ONNX."""

    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["unet", "deeplabv3plus", "segformer"],
        help="Type of model to convert",
    )
    parser.add_argument(
        "--num-classes", type=int, required=True, help="Number of segmentation classes"
    )
    parser.add_argument("--output-path", type=str, required=True, help="Path to save ONNX model")
    parser.add_argument("--weights-path", type=str, help="Path to PyTorch weights file")

    # Input shape arguments
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for conversion")
    parser.add_argument("--channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--height", type=int, default=1024, help="Input height")
    parser.add_argument("--width", type=int, default=1024, help="Input width")

    # Model-specific arguments
    parser.add_argument(
        "--encoder-name",
        type=str,
        default="resnet34",
        help="Encoder name for U-Net/DeepLabV3+ (e.g., resnet34, resnet50)",
    )
    parser.add_argument(
        "--encoder-weights",
        type=str,
        default="imagenet",
        help="Encoder weights for U-Net/DeepLabV3+",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="Backbone for SegFormer",
    )

    # Optimization arguments
    parser.add_argument(
        "--enable-optimization", action="store_true", help="Enable ONNX model optimization"
    )
    parser.add_argument(
        "--enable-fp16",
        action="store_true",
        help="Convert ONNX model to FP16 precision for smaller size and faster inference",
    )
    parser.add_argument(
        "--enable-benchmarking", action="store_true", help="Enable model performance benchmarking"
    )
    parser.add_argument("--benchmark-runs", type=int, default=50, help="Number of benchmark runs")

    parser.add_argument("--opset-version", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--skip-test", action="store_true", help="Skip inference testing")

    args = parser.parse_args()

    input_shape = (args.batch_size, args.channels, args.height, args.width)
    converter = ONNXExporter(
        model_type=args.model_type, num_classes=args.num_classes, input_shape=input_shape
    )
    model_kwargs = {
        "encoder_name": args.encoder_name,
        "encoder_weights": args.encoder_weights,
        "backbone": args.backbone,
        "in_channels": args.channels,
        "pretrained": True,
        "ignore_index": 255,
    }
    logger.info("Starting model conversion process")
    results = converter.convert_and_test(
        output_path=args.output_path,
        weights_path=args.weights_path,
        enable_optimization=args.enable_optimization,
        enable_fp16=args.enable_fp16,
        enable_benchmarking=args.enable_benchmarking,
        benchmark_runs=args.benchmark_runs,
        **model_kwargs,
    )

    print("=" * 50)
    print("CONVERSION RESULTS")
    print("=" * 50)
    print(f"Model type: {args.model_type}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Input shape: {input_shape}")
    print(f"Output path: {args.output_path}")
    print("-" * 50)
    print(f"Conversion success: {'OK' if results['conversion_success'] else 'FAIL'}")
    print(f"Optimization Success: {'OK' if results['optimization_success'] else 'FAIL'}")
    print(f"FP16 Conversion: {'OK' if results['fp16_success'] else 'FAIL'}")
    print(f"Validation success: {'OK' if results['validation_success'] else 'FAIL'}")
    print(f"Inference success: {'OK' if results['inference_success'] else 'FAIL'}")
    if results["max_diff"] is not None:
        print(f"Max difference: {results['max_diff']:.2e}")
    print("=" * 50)

    if results["conversion_success"] and results["validation_success"]:
        logger.info("Conversion completed successfully!")
        sys.exit(0)
    else:
        logger.error("Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
