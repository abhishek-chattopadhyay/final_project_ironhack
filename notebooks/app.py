import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names from the training folder
@st.cache_data
def get_class_names(data_path="../data/cv_images"):
    return sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])

class_names = get_class_names()

# Image preprocessing (should match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Normalize to [-1, 1]
])

# Load the trained model
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(torch.load("../models/best_model_basic.pt", map_location=device))
    model.eval()
    return model.to(device)

model = load_model()

# Streamlit UI
st.title("Marine Life Detector")
st.write("Upload a `.jpg` or `.png` image to find the category of the marine life:")

uploaded_file = st.file_uploader("Drag and drop or click to upload", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=False)

    # Preprocess
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = output.max(1)
        predicted_class = class_names[predicted.item()]

    st.markdown(f"### 🧠 Your species is: `{predicted_class.capitalize()}`")
