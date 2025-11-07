# 🌸 Flower Classification App (PyTorch + Streamlit)

This project classifies **5 types of flowers** using a pretrained MobileNetV2 model and deploys a simple **Streamlit web app** for predictions.

| Class Name |
|-----------|
| Daisy |
| Dandelion |
| Rose |
| Sunflower |
| Tulip |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | **90.3%** |

Model trained for 2 epochs only (backbone frozen) and still achieved strong results.

---

## ⚙️ Tech Stack
- Python
- PyTorch + Torchvision (MobileNetV2 pretrained)
- Streamlit
- Virtual Environment (on macOS)

---

## 📁 Dataset
Used the famous Flowers Dataset (Kaggle styled).  
Data was auto-split into **train/test** using `split.py`.

Folder looks like:

data/
└── train/
└── test/


---

## 🚀 How to run locally

### 1) create venv & install requirements
```bash
source .venv/bin/activate
pip install -r requirements.txt


2) train the model
python3 train.py


after training → model.pth will be saved.

3) run app
streamlit run app.py


Upload a flower image → prediction will appear instantly.