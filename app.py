import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------------------------------------
# CLASSES (fix order)
# CLASS NAMES AUTO FROM YOUR DATA FOLDER
from torchvision.datasets import ImageFolder
temp = ImageFolder("data/train")   # EXACT your training folder
CLASSES = temp.classes

# -------------------------------------

# title
st.markdown("<h1 style='text-align:center;color:#FF7A00;'>🌸 Flower Classifier App</h1>", unsafe_allow_html=True)

# load model
@st.cache_resource
def load_model():
    device = "cpu"
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# transform
tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# UI
st.write("Upload flower image and I will identify its category 🌼")

img_file = st.file_uploader("Upload Image", type=['jpg','jpeg','png'])

if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # predict
    im = tfms(img).unsqueeze(0)
    with torch.no_grad():
        preds = model(im)[0]
        sm = torch.softmax(preds, dim=0)
        idx = torch.argmax(sm).item()
        flower_name = CLASSES[idx]
        confidence = sm[idx].item()*100

    st.markdown(f"<h2 style='color:#009900;text-align:center;'>Prediction : {flower_name}</h2>", 
                unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align:center;'>Confidence : {confidence:.2f}%</h4>")
