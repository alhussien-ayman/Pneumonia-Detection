import os
import base64
import joblib
import numpy as np
import cv2
from skimage.feature import hog, graycomatrix, graycoprops
from skimage import exposure as sk_exposure
from flask import Flask, request, jsonify, render_template
import sys

sys.path.insert(0, os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'Frontend', 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), '..', 'Frontend', 'static')
)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'dcm'}

models_info = {}
global_preprocessing = {
    'raw_scaler': None,
    'pca_scaler': None,
    'pca_transformer': None,
    'eval_data': {}
}

# --- Feature Extraction Pipeline ---
def create_lung_mask(image_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(image_gray)

    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    ret, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    masked_img = cv2.bitwise_and(image_gray, image_gray, mask=mask)
    return masked_img, mask

def extract_hog_features(image):
    img_resized = cv2.resize(image, (128, 128))
    fd = hog(img_resized, orientations=9, pixels_per_cell=(8, 8),
             cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
    return fd

def extract_glcm_features(image):
    img_resized = cv2.resize(image, (128, 128))
    img_binned = (img_resized / 8).astype(np.uint8)
    glcm = graycomatrix(img_binned, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=32, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()

    return np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])

def img_to_b64(img_gray):
    """Encode a uint8 numpy image as a base64 PNG data URI."""
    _, buf = cv2.imencode('.png', img_gray)
    return 'data:image/png;base64,' + base64.b64encode(buf).decode('utf-8')


def process_image_pipeline(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, []

    img = cv2.resize(img, (256, 256))
    steps = []

    # Step 1 — Original grayscale
    steps.append({'label': 'Original (Grayscale)', 'image': img_to_b64(img)})

    # Step 2 — CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(img)
    steps.append({'label': 'CLAHE Enhanced', 'image': img_to_b64(equalized)})

    # Step 3 — Otsu mask
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    steps.append({'label': 'Otsu Mask', 'image': img_to_b64(mask)})

    # Step 4 — Lung region isolated
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    steps.append({'label': 'Lung Region', 'image': img_to_b64(masked_img)})

    # Step 5 — HOG feature map
    img128 = cv2.resize(masked_img, (128, 128))
    _, hog_img = hog(img128, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    hog_rescaled = sk_exposure.rescale_intensity(hog_img, in_range=(0, 10))
    hog_uint8 = (hog_rescaled * 255).astype(np.uint8)
    steps.append({'label': 'HOG Features', 'image': img_to_b64(cv2.resize(hog_uint8, (256, 256)))})

    hog_feats = extract_hog_features(masked_img)
    glcm_feats = extract_glcm_features(masked_img)

    return hog_feats, glcm_feats, steps


# --- Model Loading ---
def load_models():
    global models_info, global_preprocessing
    models_info = {}
    
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(models_dir):
        print("WARNING: models directory not found.")
        return

    # Load global preprocessing objects
    try:
        if os.path.exists(os.path.join(models_dir, 'raw_scaler.pkl')):
            global_preprocessing['raw_scaler'] = joblib.load(os.path.join(models_dir, 'raw_scaler.pkl'))
        if os.path.exists(os.path.join(models_dir, 'pca_scaler.pkl')):
            global_preprocessing['pca_scaler'] = joblib.load(os.path.join(models_dir, 'pca_scaler.pkl'))
        if os.path.exists(os.path.join(models_dir, 'pca_transformer.pkl')):
            global_preprocessing['pca_transformer'] = joblib.load(os.path.join(models_dir, 'pca_transformer.pkl'))
        if os.path.exists(os.path.join(models_dir, 'eval_data.pkl')):
            global_preprocessing['eval_data'] = joblib.load(os.path.join(models_dir, 'eval_data.pkl'))
        print("DONE: Loaded global preprocessing objects.")
    except Exception as e:
        print(f"WARNING: Could not load global preprocessing objects: {e}")

    # Load individual models
    for filename in os.listdir(models_dir):
        if filename.endswith('_model.pkl'):
            model_id = filename.replace('_model.pkl', '')
            model_path = os.path.join(models_dir, filename)
            
            try:
                model = joblib.load(model_path)
                
                # Fetch eval data
                metrics = global_preprocessing['eval_data'].get(model_id, {
                    'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0
                })
                
                name_map = {
                    'final_rf':  'Balanced RF (PCA Pipeline)',
                    'svm_clf5':  'SVC (PCA Pipeline)',
                    'raw_rf':    'Balanced RF (Raw Features)',
                    'raw_svc':   'SVC (Raw Features)',
                }
                display_name = name_map.get(model_id, model_id.replace('_', ' ').title())
                
                # Determine if model needs PCA
                requires_pca = ('final' in model_id.lower() or 'best' in model_id.lower() or 'svm_clf5' in model_id.lower())
                
                models_info[model_id] = {
                    'model': model,
                    'requires_pca': requires_pca,
                    'metrics': metrics,
                    'display_name': display_name
                }
                print(f"DONE: Loaded {model_id} - Requires PCA: {requires_pca}")
            except Exception as e:
                print(f"ERROR: Could not load {model_id}: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Endpoints ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    available = []
    for m_id, m_data in models_info.items():
        available.append({
            'id': m_id,
            'name': m_data['display_name'],
            'accuracy': round(m_data['metrics'].get('accuracy', 0.0) * 100, 1)
        })
    # Sort by accuracy descending so the best model is first (and default in the UI)
    available.sort(key=lambda x: x['accuracy'], reverse=True)
    return jsonify(available)

@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    if not models_info:
        return jsonify({'error': 'No models are currently loaded on the server. Please provide the models in Backend/models/'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a chest X-ray image.'}), 400

    model_id = request.form.get('model_id')
    if not model_id or model_id not in models_info:
        model_id = list(models_info.keys())[0]

    selected = models_info[model_id]
    model = selected['model']
    requires_pca = selected['requires_pca']

    try:
        file_bytes = file.read()
        hog_feats, glcm_feats, pipeline_steps = process_image_pipeline(file_bytes)

        if hog_feats is None:
            return jsonify({'error': 'Could not process image.'}), 400
            
        hog_feats = hog_feats.reshape(1, -1)
        glcm_feats = glcm_feats.reshape(1, -1)
        
        if requires_pca:
            pca = global_preprocessing.get('pca_transformer')
            scaler = global_preprocessing.get('pca_scaler')
            if pca is None or scaler is None:
                return jsonify({'error': 'PCA transformer or scaler missing for this model.'}), 500
                
            hog_pca = pca.transform(hog_feats)
            combined_feats = np.hstack([hog_pca, glcm_feats])
            scaled_feats = scaler.transform(combined_feats)
        else:
            scaler = global_preprocessing.get('raw_scaler')
            if scaler is None:
                return jsonify({'error': 'Raw scaler missing for this model.'}), 500
                
            combined_feats = np.hstack([hog_feats, glcm_feats])
            scaled_feats = scaler.transform(combined_feats)

        prediction = model.predict(scaled_feats)[0]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(scaled_feats)[0]
            confidence = round(float(max(proba)) * 100, 2)
        else:
            # SVM trained without probability=True — confidence is not available
            confidence = None

        label = 'PNEUMONIA' if prediction == 1 else 'NORMAL'

        return jsonify({
            'prediction':     label,
            'confidence':     confidence,
            'model_used':     selected['display_name'],
            'pipeline_steps': pipeline_steps,
            'status':         'success'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
@app.route('/api/metrics', methods=['GET'])
def metrics():
    model_id = request.args.get('model_id')
    if not models_info:
        return jsonify({
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0
        })
        
    if not model_id or model_id not in models_info:
        model_id = list(models_info.keys())[0]
        
    eval_data = models_info[model_id]['metrics']
    
    split_recalls = {
        'svm_clf5': {'normal': 66.0, 'pneumonia': 86.0},
        'final_rf': {'normal': 69.0, 'pneumonia': 94.0},
        'raw_rf':   {'normal': 76.0, 'pneumonia': 96.0},
        'raw_svc':  {'normal': 86.0, 'pneumonia': 88.0}
    }
    recalls = split_recalls.get(model_id, {'normal': round(eval_data.get('recall', 0) * 100, 1), 'pneumonia': round(eval_data.get('recall', 0) * 100, 1)})
    
    return jsonify({
        'accuracy':  round(eval_data.get('accuracy',  0) * 100, 1),
        'precision': round(eval_data.get('precision', 0) * 100, 1),
        'recall_normal': recalls['normal'],
        'recall_pneumonia': recalls['pneumonia'],
        'f1_score':  round(eval_data.get('f1_score',  0) * 100, 1),
    })

if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)