# 🌸 Flower Classification App (PyTorch + Streamlit)

A machine learning project that classifies **5 types** of flowers using **MobileNetV2**.

### Classes
- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

### Model
- Architecture: MobileNetV2 (ImageNet pretrained)
- Freeze: Backbone frozen
- Train only classifier ⚡

### Dataset
flowers folder → split → 80% train / 20% test automatically

### Run training
```bash
source .venv/bin/activate
python3 train.py
