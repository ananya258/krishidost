# 🌿 KrishiDost — Tomato Leaf Disease Detector
## 🌐 Live Demo
👉 [Try KrishiDost here](https://ananya0-krishidost.hf.space)
KrishiDost ("Farmer's Friend") is an AI-powered web app that helps farmers identify tomato leaf diseases instantly from a photo and get actionable treatment advice.

## 🎯 Problem
Tomato crop diseases cause massive losses for Indian farmers every year. Early detection is critical but access to agricultural experts is limited, especially in rural areas.

## 💡 Solution
Upload a photo of a tomato leaf → KrishiDost identifies the disease in seconds and provides:
- Disease name and confidence score
- Severity level
- Cause of the disease
- Immediate action to take
- Organic and chemical treatment options
- Prevention tips for next season

## 🔬 Model
- Architecture: MobileNetV2 (Transfer Learning)
- Dataset: PlantVillage (18,345 tomato leaf images)
- Classes: 10 (9 diseases + healthy)
- Validation Accuracy: **93.54%**
- Training: PyTorch, trained on Kaggle GPU

## 🛠️ Tech Stack
- Python
- PyTorch & TorchVision
- Streamlit
- MobileNetV2 (pretrained on ImageNet)

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Diseases Detected
1. Bacterial Spot
2. Early Blight
3. Late Blight
4. Leaf Mold
5. Septoria Leaf Spot
6. Spider Mites
7. Target Spot
8. Tomato Yellow Leaf Curl Virus
9. Tomato Mosaic Virus
10. Healthy

## 👩‍💻 Built by
Ananya Tiwari : BTech CSE, KIIT University