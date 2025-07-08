# Vehicle AI Damage Inspection

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-purple.svg)](https://lightning.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Machine learning pipeline for training and exporting **instance segmentation** models for damage detection and **semantic segmentation** models for car parts segmentation. Built with PyTorch Lightning and Hydra for scalable and configurable training and deployment.

---

## Key Features

### **Dual Tasks**
- **Instance Segmentation**: Damage detection and localization using Mask2Former
- **Semantic Segmentation**: Complete car parts segmentation with multiple SOTA models

### **State-of-the-Art Models**
- **Mask2Former**: Advanced transformer-based segmentation for both tasks
- **SegFormer**: Efficient transformer architecture for semantic segmentation  
- **DeepLabV3+**: Proven CNN architecture with atrous convolutions
- **U-Net**: Classic encoder-decoder with skip connections

### **Production Ready**
- **ONNX Export**: Convert semantic models for optimized inference
- **Docker Support**: Containerized environment with CUDA support
- **Comprehensive Testing**: Full test suite for all components
- **Hydra Configuration**: Flexible experiment management

### **Advanced Training Pipeline**
- **PyTorch Lightning**: Scalable training with automatic mixed precision
- **Rich Data Augmentation**: Albumentations-based preprocessing
- **Multiple Metrics**: mAP for instance, IoU/Accuracy for semantic tasks
- **MLFlow Integration**: Experiment tracking and model versioning (other loggers can also be easily configured)

---

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- Docker (optional)
- Poetry

### Installation

0. **Install the Poetry, if not installed**
```bash
python3 -m pip install pipx
pipx install poetry
pipx ensurepath
```

1. **Clone the repository**
```bash
git clone https://github.com/rizvansky/vehicle-ai-damage-inspection.git
cd vehicle-ai-damage-inspection
```

2. **Install dependencies**
```bash
poetry install
```

### Docker Setup

Build and run with Docker for a consistent environment:

```bash
# Build the container
docker build -t vehicle-ai-damage-inspection .

# For Mac OS with ARM
docker buildx build --platform linux/arm64 -t vehicle-ai-damage-inspection .

# Run with GPU support
docker run --gpus all -v $(pwd):/code -it vehicle-ai-damage-inspection bash
```

## Exploratory Data Analysis
- [EDA on damage instance segmentation (using proprietary dataset)](notebooks/01-eda-damage-detection.ipynb)
- [EDA on car parts semantic segmentation (using proprietary dataset)](notebooks/02-eda-car-parts-segmentation.ipynb)

## Usage

Firstly, activate Poetry environment:
```bash
poetry env activate
source <venv-path>/bin/activate
```

Or add `poetry run` before commands execution.

### Training Models

#### Instance Segmentation (Damage Detection and Segmentation)
```bash
# Train Mask2Former for damage detection
python src/train.py experiment=instance_mask2former

# Or with custom parameters
python src/train.py \
  data=instance_damage_detection \
  model=mask2former_instance \
  trainer.max_epochs=50 \
  data.batch_size=4
```

#### Semantic Segmentation (Car Parts Segmentation)
```bash
# Train different semantic models
python src/train.py experiment=segformer
python src/train.py experiment=unet  
python src/train.py experiment=deeplabv3plus
python src/train.py experiment=semantic_mask2former
```

### ONNX Export

Convert trained semantic segmentation models to ONNX for deployment:

```bash
# Export U-Net model
python scripts/export_to_onnx.py \
  --model-type unet \
  --num-classes 32 \
  --output-path models/unet_model.onnx \
  --weights-path checkpoints/unet_best.ckpt \
  --enable-optimization \
  --enable-fp16

# Export SegFormer model  
python scripts/export_to_onnx.py \
  --model-type segformer \
  --num-classes 32 \
  --output-path models/segformer_model.onnx \
  --backbone nvidia/mit-b5 \
  --enable-benchmarking
```
- **Mask2Former** models (both for instance and semantic segmentation) are not supported for ONNX export due to architectural challenges.
- **SegFormer** model is exported via `optimum.onnxruntime`.
- **U-Net** and **DeepLabV3+** models are exported in a standard way using `torch.onnx.export`.


### Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/models/                    # Model-specific modules testing
pytest tests/data/                      # Data-specific modules testing  
pytest tests/test_export.py             # ONNX export testing
```

## Project Structure

```
vehicle-ai-damage-inspection/
├── configs/                                <- Hydra configurations
│   ├── callbacks/                              <- Callbacks
│   ├── data/                                   <- Lightning datamodules
│   ├── experiment/                             <- Experiments
│   ├── extras/                                 <- Extra training options
│   ├── hydra/                                  <- Base Hydra configuration
│   ├── logger/                                 <- Loggers
│   ├── model/                                  <- Lightning modules
│   ├── paths/                                  <- Paths configuration, including default logging dir etc.
│   ├── trainer/                                <- Trainer configurations
│   └── train.yaml                              <- Main train configuration file
│
├── notebooks/                              <- Jupyter Notebooks
│   ├── 01-eda-damage-detection.ipynb           <- Damage instance segmentation dataset EDA
│   └── 02-eda-car-parts-segmentation.ipynb     <- Car parts segmentation dataset EDA
│
├── scripts/                                <- Scripts, not related to training process
│   └── export_to_onnx.py                       <- Export models to ONNX
│
├── src/                                    <- Main source code
│   ├── data/                                   <- Data-specific modules
│   │   ├── components/                             <- Datasets implementations
│   │   ├── instance_datamodule.py                  <- Datamodule for instance segmentation task
│   │   └── semantic_datamodule.py                  <- Datamodule for semantic segmentation task
│   ├── export/                                 <- Export utilities
│   │   └── onnx_exporter.py                        <- ONNX exporter
│   ├── models/                                 <- Models implementations
│   │   ├── components/
│   │   │   ├── instance/                               <- Instance segmentation
│   │   │   └── semantic/                               <- Semantic segmentation
│   │   ├── instance_module.py                      <- Lightning module for instance segmentation task
│   │   └── semantic_module.py                      <- Lightning module for semantic segmentation task
│   ├── utils/                                  <- Utility functions
│   └── train.py                                <- Main training script
│
├── tests/                                  <- Tests
│   ├── data/                                   <- Data-specific modules
│   ├── models/                                 <- Model-specific modules
│   └── test_export.py                          <- Export functionality
│
├── .dockerignore
├── .gitignore
├── .project-root                           <- Flag file to indicating root directory when launching code
├── Dockerfile
├── poetry.lock                             <- Poetry configuration file
├── pyproject.toml                          <- Poetry configuration file
└── README.md
```

## Configuration System

The project uses **Hydra** for configuration management:

### Dataset Configuration
```yaml
# configs/data/instance_damage_detection.yaml
_target_: src.data.instance_datamodule.InstanceDataModule
images_dir: /path/to/images
train_annotations_path: /path/to/train.json
batch_size: 8
num_workers: 8

train_transform:
  transforms:
    - _target_: albumentations.RandomScale
      scale_limit: [-0.5, 1.0]
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    # ... more augmentations
```

### Model Configuration  
```yaml
# configs/model/mask2former_instance.yaml
_target_: src.models.instance_module.InstanceSegmentationModule
net:
  _target_: src.models.components.instance.mask2former.Mask2FormerInstance
  num_classes: 19
  backbone: facebook/mask2former-swin-large-coco-instance
  pretrained: true

optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-4
  weight_decay: 0.0
```

### Experiment Presets
```yaml
# configs/experiment/segformer.yaml
defaults:
  - override /data: semantic_vehicle_parts
  - override /model: segformer
  - override /trainer: default

tags: ["segformer", "semantic", "car_parts_segmentation"]
seed: 28

trainer:
  max_epochs: 50
  precision: 32
```

## Data Formats

### Instance Segmentation (COCO Format)
```json
{
  "images": [{"id": 1, "file_name": "car_001.jpg", "width": 1024, "height": 768}],
  "annotations": [
    {
      "id": 1,
      "image_id": 1, 
      "category_id": 1,
      "bbox": [x, y, width, height],
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 1234.5
    }
  ],
  "categories": [{"id": 1, "name": "scratch"}, {"id": 2, "name": "dent"}]
}
```

### Semantic Segmentation
```
data/
├── train/
│   ├── images/
│   │   ├── car_001.jpg
│   │   └── car_002.jpg
│   └── masks/
│       ├── car_001.png    # Pixel values = class IDs
│       └── car_002.png
├── val/
└── test/
```

## Training & Evaluation

### Key Metrics

**Instance Segmentation**:
- **mAP (mask)**: Mean Average Precision for segmentation masks
- **mAP@50**: mAP at IoU threshold 0.5  
- **mAP@75**: mAP at IoU threshold 0.75
- **Per-class mAP**: Individual performance per damage type

**Semantic Segmentation**:
- **mIoU**: Mean Intersection over Union
- **Pixel Accuracy**: Overall pixel classification accuracy
- **Per-class IoU**: Individual performance per car part


### Data Augmentation

**Instance Segmentation**:
- Large Scale Jittering (LSJ)
- Random crops with bbox-safe cropping
- Horizontal flip, rotation, perspective transform
- Color jittering, brightness/contrast adjustment
- Blur and noise augmentations

**Semantic Segmentation**:
- Resize and random crops
- Flip and rotation augmentations  
- Brightness, contrast, saturation adjustment
- Gaussian noise and blur

## Advanced Features

### ONNX Export Capabilities
- **Model Optimization**: Using ONNXSim for graph optimization
- **FP16 Conversion**: Reduce model size by 50%
- **Performance Benchmarking**: Automated inference speed testing
- **Validation**: Ensure ONNX output matches PyTorch

### Flexible Architecture
- **Modular Design**: Easy to add new models and datasets
- **Plugin System**: Custom callbacks and loggers
- **Configuration Override**: Change any parameter via command line

## License

This project is licensed under the MIT License.
