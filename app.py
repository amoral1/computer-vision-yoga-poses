%%writefile app.py
import streamlit as st
import numpy as np
import json
from PIL import Image
import onnxruntime as ort
from ultralytics import YOLO

st.set_page_config(page_title="Yoga Pose Classifier", layout="wide")

MODEL_SAVE_DIR = './models'
IMG_SIZE = 160

@st.cache_resource
def load_assets():
    models = {
        'Advanced CNN':   ort.InferenceSession(f'{MODEL_SAVE_DIR}/yoga_advanced.onnx'),
        'VGG16 Transfer': ort.InferenceSession(f'{MODEL_SAVE_DIR}/yoga_vgg16.onnx'),
    }
    with open(f'{MODEL_SAVE_DIR}/class_names.json') as f:
        class_names = json.load(f)
    return models, class_names

@st.cache_resource
def load_yolo():
    return YOLO('yolov8n-pose.pt')

def preprocess(img: Image.Image, size=IMG_SIZE):
    img = img.convert('RGB').resize((size, size))
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def top_k_predictions(session, x, class_names, k=5):
    input_name = session.get_inputs()[0].name
    probs = session.run(None, {input_name: x})[0][0]
    top_idx = probs.argsort()[-k:][::-1]
    return [(class_names[i], float(probs[i])) for i in top_idx]

# ── UI ──────────────────────────────────────────────────────────────────────
st.title("🧘 Yoga Pose Classifier")
st.caption("Custom CNN vs. VGG16 transfer learning, trained on 107 yoga poses")

models, class_names = load_assets()
yolo = load_yolo()

with st.sidebar:
    st.header("Settings")
    selected_models = st.multiselect(
        "Compare which models?",
        list(models.keys()),
        default=list(models.keys())
    )
    top_k = st.slider("Top-K predictions to show", 1, 10, 5)

uploaded = st.file_uploader("Upload a yoga pose image", type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    img_arr = np.array(img)

    results = yolo(img_arr)
    annotated = results[0].plot()[..., ::-1]

    boxes = results[0].boxes
    person_boxes = [b for b in boxes if int(b.cls) == 0] if boxes is not None else []
    if person_boxes:
        box = person_boxes[0].xyxy[0].cpu().numpy().astype(int)
        img_for_classifier = img.crop((box[0], box[1], box[2], box[3]))
        crop_caption = "Cropped to detected person"
    else:
        img_for_classifier = img
        crop_caption = "No person detected — using full image"

    col_img, col_preds = st.columns([1, 2])

    with col_img:
        st.image(img, caption="Input", use_container_width=True)
        st.image(annotated, caption="YOLO pose detection", use_container_width=True)
        st.image(img_for_classifier, caption=crop_caption, use_container_width=True)

    with col_preds:
        x = preprocess(img_for_classifier)
        for name in selected_models:
            st.subheader(name)
            preds = top_k_predictions(models[name], x, class_names, k=top_k)
            for pose, prob in preds:
                st.write(f"**{pose}** — {prob*100:.1f}%")
                st.progress(min(prob, 1.0))
            st.divider()
else:
    st.info("Upload an image to see predictions from each  model.")
