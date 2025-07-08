from tempfile import TemporaryDirectory

import pytest

from src.export.onnx_exporter import ONNXExporter


@pytest.mark.parametrize("model_name", ["unet", "deeplabv3plus", "segformer"])
def test_export(model_name: str) -> None:
    """Text export to ONNX of a model with name `"model_name"`.
        Checks the following processes:
            - Conversion to ONNX
            - Model optimization
            - Conversion to FP16
            - Model validation
            - Model inference

    Args:
        model_name (str): Model name. One of: `"unet"`, `"deeplabv3plus"`, `"segformer"`.
    """

    converter = ONNXExporter(model_name, 32, (1, 3, 512, 512))
    model_kwargs = {
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "backbone": "nvidia/segformer-b0-finetuned-ade-512-512",
        "in_channels": 3,
        "pretrained": False,
        "ignore_index": 255,
    }
    with TemporaryDirectory() as tmp_dir:
        results = converter.convert_and_test(
            output_path=f"{tmp_dir}/model.onnx",
            weights_path=None,
            enable_optimization=True,
            enable_fp16=True,
            enable_benchmarking=False,
            **model_kwargs,
        )

    assert results["conversion_success"]
    assert results["optimization_success"]
    assert results["fp16_success"]
    assert results["validation_success"]
    assert results["inference_success"]
