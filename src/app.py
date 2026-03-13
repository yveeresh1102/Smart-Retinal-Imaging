import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# Page config
st.set_page_config(page_title="Smart Retinal Imaging", layout="wide")

st.title("Smart Retinal Imaging: AI-based Vascular Health Assessment")

# Load model
@st.cache_resource
def load_model():

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    model.load_state_dict(
        torch.load("retinal_segmentation_model.pth", map_location="cpu")
    )

    model.eval()

    return model

model = load_model()

# CLAHE preprocessing
def apply_clahe_rgb(image):

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    cl = clahe.apply(l)

    merged = cv2.merge((cl,a,b))

    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return enhanced


# Feature extraction
def extract_features(image, mask):

    mask = (mask > 0.5).astype(np.uint8)

    vessel_density = np.sum(mask) / mask.size

    width = np.mean(cv2.distanceTransform(mask, cv2.DIST_L2, 5))

    tortuosity = vessel_density * 20

    return {
        "vessel_density": vessel_density,
        "mean_width": width,
        "mean_tortuosity": tortuosity
    }


# Health classification
def classify_health(features):

    score = 0

    if features["vessel_density"] > 0.09:
        score += 1

    if features["mean_tortuosity"] > 1.7:
        score += 1

    if features["mean_width"] > 3.2:
        score += 1

    if score == 0:
        return "Low Risk"
    elif score == 1:
        return "Medium Risk"
    else:
        return "High Risk"


# Upload image
uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg","png","tif"])

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Original Image", use_container_width=True)

    # Preprocess
    image_clahe = apply_clahe_rgb(image)

    img_resized = cv2.resize(image_clahe, (512,512))

    img_tensor = torch.tensor(img_resized/255.0).permute(2,0,1).unsqueeze(0).float()

    # Segmentation
    with torch.no_grad():

        pred = torch.sigmoid(model(img_tensor))

    mask = pred[0][0].numpy()

    mask_binary = (mask > 0.65).astype(np.uint8)

    # Show segmentation
    st.image(mask_binary*255, caption="Vessel Segmentation")

    # Overlay
    overlay = image.copy()

    overlay = cv2.resize(overlay, (512,512))

    overlay[mask_binary==1] = [255,0,0]

    st.image(overlay, caption="Overlay")

    # Feature extraction
    features = extract_features(img_resized, mask_binary)

    # Classification
    health = classify_health(features)

    st.success(f"Vascular Health Status: {health}")
