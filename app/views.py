import os
import json
import numpy as np
from django.shortcuts import render, redirect
from .models import LeafScan

# ── Lazy TensorFlow import (faster Django startup) ─────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(_file_))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_model')
MODEL_PATH  = os.path.join(MODEL_DIR, 'plant_disease_model.h5')
LABELS_PATH = os.path.join(MODEL_DIR, 'class_names.json')

model = None
class_names = {}
MODEL_LOADED = False

try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image

    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABELS_PATH) as f:
            class_names = json.load(f)   # {"0": "Tomato___Healthy", ...}
        MODEL_LOADED = True
        print(" Plant disease model loaded successfully")
    else:
        print("  Model files not found. Run train_model.py first.")
except Exception as e:
    print(f"  Could not load model: {e}")


# ── Disease info: treatment & prevention tips ───────────────────────────────────
DISEASE_INFO = {
    "Healthy": {
        "description": "The leaf shows no visible signs of disease.",
        "treatment": "No action needed. Continue regular watering, sunlight, and fertilization.",
    },
    "Bacterial_Spot": {
        "description": "Bacterial infection causing dark, water-soaked spots on leaves.",
        "treatment": "Apply copper-based bactericides. Avoid overhead watering. Remove infected leaves.",
    },
    "Early_Blight": {
        "description": "Fungal disease causing brown concentric-ring spots on older leaves.",
        "treatment": "Apply fungicide (chlorothalonil/mancozeb). Rotate crops. Remove infected debris.",
    },
    "Late_Blight": {
        "description": "Aggressive fungal-like disease causing dark, irregular lesions that spread fast.",
        "treatment": "Apply fungicide immediately. Destroy infected plants to prevent spread. Improve drainage.",
    },
}

DEFAULT_INFO = {
    "description": "Disease detected. Consult an agricultural expert for an exact treatment plan.",
    "treatment": "Isolate the affected plant, remove damaged leaves, and consult local agri-extension services.",
}


def clean_label(raw_label):
    """ 'Tomato___Late_Blight' -> ('Tomato', 'Late Blight') """
    parts = raw_label.split('_')
    plant = parts[0].replace('_', ' ')
    condition = parts[1].replace('', ' ') if len(parts) > 1 else raw_label.replace('', ' ')
    return plant, condition


def get_severity(confidence, is_healthy):
    if is_healthy:
        return "None"
    if confidence < 60:
        return "Low"
    elif confidence < 85:
        return "Moderate"
    else:
        return "High"


# ── Views ──────────────────────────────────────────────────────────────────────

def home(request):
    total   = LeafScan.objects.count()
    healthy = LeafScan.objects.filter(is_healthy=True).count()
    diseased = total - healthy
    context = {
        'total': total,
        'healthy': healthy,
        'diseased': diseased,
        'model_loaded': MODEL_LOADED,
    }
    return render(request, 'detection/home.html', context)


def predict(request):
    if request.method == 'POST' and request.FILES.get('leaf_image'):
        if not MODEL_LOADED:
            return render(request, 'detection/home.html',
                           {'error': 'Model not loaded. Run train_model.py first.'})

        try:
            import tensorflow as tf
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            from tensorflow.keras.preprocessing import image as keras_image

            uploaded_file = request.FILES['leaf_image']
            plant_name    = request.POST.get('plant_name', 'Unknown')

            # Save record first (so the image is stored on disk)
            scan = LeafScan.objects.create(
                image=uploaded_file,
                plant_name=plant_name,
                predicted_class="pending",
                confidence=0.0,
            )

            # ── Preprocess image for the model ──────────────────────────────────
            img_path = scan.image.path
            img = keras_image.load_img(img_path, target_size=(224, 224))
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # ── Predict ───────────────────────────────────────────────────────────
            preds = model.predict(img_array)[0]
            class_idx = int(np.argmax(preds))
            confidence = round(float(preds[class_idx]) * 100, 2)
            raw_label = class_names[str(class_idx)]

            is_healthy = "healthy" in raw_label.lower()
            plant_clean, condition_clean = clean_label(raw_label)
            severity = get_severity(confidence, is_healthy)

            info_key = raw_label.split('_')[-1] if '_' in raw_label else raw_label
            info = DISEASE_INFO.get(info_key, DEFAULT_INFO)

            # ── Update record with prediction ────────────────────────────────────
            scan.predicted_class = raw_label
            scan.is_healthy = is_healthy
            scan.confidence = confidence
            scan.severity = severity
            scan.save()

            context = {
                'scan': scan,
                'plant_clean': plant_clean,
                'condition_clean': condition_clean,
                'confidence': confidence,
                'is_healthy': is_healthy,
                'severity': severity,
                'description': info['description'],
                'treatment': info['treatment'],
            }
            return render(request, 'detection/result.html', context)

        except Exception as e:
            return render(request, 'detection/home.html', {'error': str(e)})

    return redirect('home')


def history(request):
    scans = LeafScan.objects.all()[:50]
    return render(request, 'detection/history.html', {'scans': scans})


def about(request):
    return render(request, 'detection/about.html')