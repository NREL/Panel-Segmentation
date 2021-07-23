![Logo](https://storage.googleapis.com/tf_model_garden/tf_model_garden_logo.png)

# TensorFlow Community Models

This repository provides a curated list of the GitHub repositories with machine learning models and implementations powered by TensorFlow 2.

**Note**: Contributing companies or individuals are responsible for maintaining their repositories.

## Computer Vision

### Image Recognition

| Model | Paper | Features | Maintainer |
|-------|-------|----------|------------|
| [DenseNet 169](https://github.com/IntelAI/models/tree/master/benchmarks/image_recognition/tensorflow/densenet169) | [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993) | • FP32 Inference | [Intel](https://github.com/IntelAI) |
| [Inception V3](https://github.com/IntelAI/models/tree/master/benchmarks/image_recognition/tensorflow/inceptionv3) | [Rethinking the Inception Architecture<br/>for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf) | • Int8 Inference<br/>• FP32 Inference | [Intel](https://github.com/IntelAI) |
| [Inception V4](https://github.com/IntelAI/models/tree/master/benchmarks/image_recognition/tensorflow/inceptionv4) | [Inception-v4, Inception-ResNet and the Impact<br/>of Residual Connections on Learning](https://arxiv.org/pdf/1602.07261) | • Int8 Inference<br/>• FP32 Inference | [Intel](https://github.com/IntelAI) |
| [MobileNet V1](https://github.com/IntelAI/models/tree/master/benchmarks/image_recognition/tensorflow/mobilenet_v1) | [MobileNets: Efficient Convolutional Neural Networks<br/>for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861) | • Int8 Inference<br/>• FP32 Inference | [Intel](https://github.com/IntelAI) |
| [ResNet 101](https://github.com/IntelAI/models/tree/master/benchmarks/image_recognition/tensorflow/resnet101) | [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385) | • Int8 Inference<br/>• FP32 Inference | [Intel](https://github.com/IntelAI) |
| [ResNet 50](https://github.com/IntelAI/models/tree/master/benchmarks/image_recognition/tensorflow/resnet50) | [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385) | • Int8 Inference<br/>• FP32 Inference | [Intel](https://github.com/IntelAI) |
| [ResNet 50v1.5](https://github.com/IntelAI/models/tree/master/benchmarks/image_recognition/tensorflow/resnet50v1_5) | [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385) | • Int8 Inference<br/>• FP32 Inference<br/>• FP32 Training | [Intel](https://github.com/IntelAI) |
| [EfficientNet](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Classification/ConvNets/efficientnet) | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf) | • Automatic mixed precision<br/>• Horovod Multi-GPU training (NCCL)<br/>• Multi-node training on a Pyxis/Enroot Slurm cluster<br/>• XLA | [NVIDIA](https://github.com/NVIDIA) |

### Object Detection

| Model | Paper | Features | Maintainer |
|-------|-------|----------|------------|
| [R-FCN](https://github.com/IntelAI/models/tree/master/benchmarks/object_detection/tensorflow/rfcn) | [R-FCN: Object Detection<br/>via Region-based Fully Convolutional Networks](https://arxiv.org/pdf/1605.06409) | • Int8 Inference<br/>• FP32 Inference | [Intel](https://github.com/IntelAI) |
| [SSD-MobileNet](https://github.com/IntelAI/models/tree/master/benchmarks/object_detection/tensorflow/ssd-mobilenet) | [MobileNets: Efficient Convolutional Neural Networks<br/>for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861) | • Int8 Inference<br/>• FP32 Inference | [Intel](https://github.com/IntelAI) |
| [SSD-ResNet34](https://github.com/IntelAI/models/tree/master/benchmarks/object_detection/tensorflow/ssd-resnet34) | [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325) | • Int8 Inference<br/>• FP32 Inference<br/>• FP32 Training | [Intel](https://github.com/IntelAI) |

### Segmentation

| Model | Paper | Features | Maintainer |
|-------|-------|----------|------------|
| [Mask R-CNN](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN) | [Mask R-CNN](https://arxiv.org/abs/1703.06870) | • Automatic Mixed Precision<br/>• Multi-GPU training support with Horovod<br/>• TensorRT | [NVIDIA](https://github.com/NVIDIA) |
| [U-Net Medical Image Segmentation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/UNet_Medical) | [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) | • Automatic Mixed Precision<br/>• Multi-GPU training support with Horovod<br/>• TensorRT | [NVIDIA](https://github.com/NVIDIA) |

## Natural Language Processing

| Model | Paper | Features | Maintainer |
|-------|-------|----------|------------|
| [BERT](https://github.com/IntelAI/models/tree/master/benchmarks/language_modeling/tensorflow/bert_large) | [BERT: Pre-training of Deep Bidirectional Transformers<br/>for Language Understanding](https://arxiv.org/pdf/1810.04805) | • FP32 Inference<br/>• FP32 Training | [Intel](https://github.com/IntelAI) |
| [BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/BERT) | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805) | • Horovod Multi-GPU<br/>• Multi-node with Horovod and Pyxis/Enroot Slurm cluster<br/>• XLA<br/>• Automatic mixed precision<br/>• LAMB | [NVIDIA](https://github.com/NVIDIA) |
| [ELECTRA](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/ELECTRA) | [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/forum?id=r1xMH1BtvB) | • Automatic Mixed Precision<br/>• Multi-GPU training support with Horovod<br/>• Multi-node training on a Pyxis/Enroot Slurm cluster | [NVIDIA](https://github.com/NVIDIA) |
| [GNMT](https://github.com/IntelAI/models/tree/master/benchmarks/language_translation/tensorflow/mlperf_gnmt) | [Google’s Neural Machine Translation System:<br/>Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144) | • FP32 Inference | [Intel](https://github.com/IntelAI) |
| [Transformer-LT (Official)](https://github.com/IntelAI/models/tree/master/benchmarks/language_translation/tensorflow/transformer_lt_official) | [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) | • FP32 Inference | [Intel](https://github.com/IntelAI) |
| [Transformer-LT (MLPerf)](https://github.com/IntelAI/models/tree/master/benchmarks/language_translation/tensorflow/transformer_mlperf) | [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) | • FP32 Training | [Intel](https://github.com/IntelAI) |

## Recommendation Systems

| Model | Paper | Features | Maintainer |
|-------|-------|----------|------------|
| [Wide & Deep](https://github.com/IntelAI/models/tree/master/benchmarks/recommendation/tensorflow/wide_deep_large_ds) | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792) | • FP32 Inference<br/>• FP32 Training | [Intel](https://github.com/IntelAI) |
| [Wide & Deep](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Recommendation/WideAndDeep) | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792) | • Automatic mixed precision<br/>• Multi-GPU training support with Horovod<br/>• XLA | [NVIDIA](https://github.com/NVIDIA) |
| [DLRM](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Recommendation/DLRM) | [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/pdf/1906.00091.pdf) | • Automatic Mixed Precision<br/>• Hybrid-parallel multiGPU training using Horovod all2all<br/>• Multinode training for Pyxis/Enroot Slurm clusters<br/>• XLA<br/>• Criteo dataset preprocessing with Spark on GPU | [NVIDIA](https://github.com/NVIDIA) |

## Contributions

If you want to contribute, please review the [contribution guidelines](https://github.com/tensorflow/models/wiki/How-to-contribute).
