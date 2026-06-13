import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

st.set_page_config(page_title="KrishiDost", page_icon="🌿", layout="centered")

disease_info = {
    "Bacterial_spot": {
        "cause": "Bacterial infection (Xanthomonas campestris). Spreads through rain and wind.",
        "severity": "Moderate",
        "immediate_action": "Remove and destroy infected leaves immediately.",
        "organic_treatment": "Copper-based spray (Bordeaux mixture) every 7 days.",
        "chemical_treatment": "Streptomycin-based bactericide spray.",
        "prevention": "Avoid overhead watering. Use disease-free seeds. Rotate crops yearly."
    },
    "Early_blight": {
        "cause": "Fungal infection (Alternaria solani). Favored by warm, humid conditions.",
        "severity": "Moderate",
        "immediate_action": "Remove infected lower leaves. Improve air circulation around plants.",
        "organic_treatment": "Neem oil spray every 7 days. Baking soda solution (1 tsp per litre).",
        "chemical_treatment": "Mancozeb or Chlorothalonil fungicide spray.",
        "prevention": "Avoid wetting leaves while watering. Mulch around plants. Crop rotation."
    },
    "Late_blight": {
        "cause": "Fungal-like pathogen (Phytophthora infestans). Spreads rapidly in cool, wet weather.",
        "severity": "Severe",
        "immediate_action": "Remove and burn infected plants immediately. Do not compost them.",
        "organic_treatment": "Copper fungicide spray. Remove all affected tissue urgently.",
        "chemical_treatment": "Metalaxyl or Cymoxanil based fungicide. Act immediately.",
        "prevention": "Plant resistant varieties. Ensure good drainage. Avoid overhead irrigation."
    },
    "Leaf_Mold": {
        "cause": "Fungal infection (Passalora fulva). Thrives in high humidity above 85%.",
        "severity": "Moderate",
        "immediate_action": "Improve ventilation. Remove heavily infected leaves.",
        "organic_treatment": "Neem oil spray. Reduce humidity by spacing plants wider apart.",
        "chemical_treatment": "Chlorothalonil or Mancozeb fungicide.",
        "prevention": "Maintain humidity below 85%. Ensure good air circulation. Avoid wetting leaves."
    },
    "Septoria_leaf_spot": {
        "cause": "Fungal infection (Septoria lycopersici). Spreads through water splash and tools.",
        "severity": "Moderate",
        "immediate_action": "Remove infected lower leaves. Disinfect gardening tools after use.",
        "organic_treatment": "Copper spray or neem oil every 7-10 days.",
        "chemical_treatment": "Mancozeb or Chlorothalonil fungicide spray.",
        "prevention": "Avoid overhead watering. Mulch soil surface. Crop rotation every season."
    },
    "Spider_mites": {
        "cause": "Tiny mites (Tetranychus urticae) feeding on leaf tissue. Worse in hot dry weather.",
        "severity": "Moderate",
        "immediate_action": "Spray plants with strong stream of water to dislodge mites.",
        "organic_treatment": "Neem oil spray every 5-7 days. Introduce predatory insects if possible.",
        "chemical_treatment": "Miticide spray (Abamectin or Spiromesifen).",
        "prevention": "Maintain adequate moisture. Inspect undersides of leaves regularly."
    },
    "Target_Spot": {
        "cause": "Fungal infection (Corynespora cassiicola). Spreads in warm humid conditions.",
        "severity": "Moderate",
        "immediate_action": "Remove infected leaves. Avoid working with plants when wet.",
        "organic_treatment": "Neem oil or copper-based spray every 7 days.",
        "chemical_treatment": "Azoxystrobin or Difenoconazole fungicide.",
        "prevention": "Ensure good air circulation. Avoid overhead irrigation. Crop rotation."
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "cause": "Viral disease spread by whiteflies. No cure once infected.",
        "severity": "Severe",
        "immediate_action": "Remove and destroy infected plants immediately to prevent spread.",
        "organic_treatment": "Control whitefly population using yellow sticky traps and neem oil.",
        "chemical_treatment": "Imidacloprid insecticide to control whitefly vectors.",
        "prevention": "Use virus-resistant tomato varieties. Install insect-proof nets. Control whiteflies early."
    },
    "Tomato_mosaic_virus": {
        "cause": "Viral disease spread through contact, infected tools, and hands.",
        "severity": "Moderate to Severe",
        "immediate_action": "Remove infected plants. Wash hands thoroughly after handling plants.",
        "organic_treatment": "No cure. Focus on prevention and removing infected plants.",
        "chemical_treatment": "No chemical cure. Control aphid vectors with insecticide.",
        "prevention": "Use virus-free seeds. Disinfect tools with bleach solution. Wash hands frequently."
    },
    "healthy": {
        "cause": "No disease detected.",
        "severity": "None",
        "immediate_action": "Your plant looks healthy! Continue regular care.",
        "organic_treatment": "Maintain regular watering and fertilization schedule.",
        "chemical_treatment": "No treatment needed.",
        "prevention": "Continue good farming practices — proper spacing, crop rotation, and regular inspection."
    }
}

tomato_class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 10)
    )
    model.load_state_dict(torch.load('krishidost_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_info_key(raw_name):
    for key in disease_info.keys():
        if key.lower() in raw_name.lower() or raw_name.lower() in key.lower():
            return key
    return None

st.title("🌿 KrishiDost")
st.subheader("Tomato Leaf Disease Detector")
st.write("Upload a photo of a tomato leaf and get an instant diagnosis.")

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    if image.size[0] > 1000 or image.size[1] > 1000:
        image = image.resize((800, 800))
    st.image(image, caption="Uploaded Leaf", use_container_width=True)

    with st.spinner("Analyzing..."):
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        raw_name = tomato_class_names[predicted.item()].replace('Tomato___', '')
        disease_name = raw_name.replace('_', ' ')
        conf_pct = round(confidence.item() * 100, 2)
        info_key = get_info_key(raw_name)
        info = disease_info.get(info_key, {})

    st.markdown("---")
    st.markdown(f"## 🔍 Diagnosis: {disease_name}")
    st.markdown(f"**Confidence:** {conf_pct}%")

    severity = info.get('severity', 'Unknown')
    if severity == 'Severe':
        st.error(f"⚠️ Severity: {severity}")
    elif severity == 'None':
        st.success(f"✅ Severity: {severity}")
    else:
        st.warning(f"⚠️ Severity: {severity}")

    st.markdown("---")
    st.markdown(f"**🦠 Cause:** {info.get('cause', 'N/A')}")
    st.markdown(f"**⚡ Immediate Action:** {info.get('immediate_action', 'N/A')}")
    st.markdown(f"**🌿 Organic Treatment:** {info.get('organic_treatment', 'N/A')}")
    st.markdown(f"**💊 Chemical Treatment:** {info.get('chemical_treatment', 'N/A')}")
    st.markdown(f"**🛡️ Prevention:** {info.get('prevention', 'N/A')}")