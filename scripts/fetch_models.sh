#!/bin/bash
set -e

MODELS_DIR=/opt/yoga-app/models
mkdir -p $MODELS_DIR

BASE=https://huggingface.co/anais1/yoga-pose-models/tree/main
for f in yoga_baseline.onnx yoga_advanced.onnx yoga_vgg16.onnx class_names.json; do
    echo "→ Fetching $f"
    wget -q --show-progress -O "$MODELS_DIR/$f" "$BASE/$f"
done

echo '→ All models synced'
