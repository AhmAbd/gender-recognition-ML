import os
import numpy as np
from PIL import Image
import cv2
import cupy as cp
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import logging
import time
from tqdm import tqdm
import kagglehub
import albumentations as A
import tensorflow as tf
import random
import psutil
from google.colab import drive
from IPython.display import Image as IPImage
try:
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.preprocessing.image import img_to_array
except ImportError as e:
    raise ImportError("Failed to import tensorflow.keras.applications. Ensure TensorFlow is installed: pip install tensorflow[and-cuda]") from e

warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mount Google Drive
drive.mount('/content/drive')
output_dir = '/content/drive/MyDrive/gender_recognition'
os.makedirs(output_dir, exist_ok=True)

# Verify GPU
!nvidia-smi

# Monitor RAM
def log_memory():
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / 1024**2:.2f} MB")

# Dataset Download
path = kagglehub.dataset_download("maciejgronczynski/biggest-genderface-recognition-dataset")
logger.info(f"Dataset path: {path}")
man_dir = os.path.join(path, "faces", "man")
woman_dir = os.path.join(path, "faces", "woman")

# Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    logger.error("Failed to load face cascade classifier")
    raise Exception("Face cascade classifier not loaded")

def is_valid_face(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, minSize=(40, 40))
        return len(faces) > 0
    except Exception as e:
        logger.warning(f"Face detection failed for {img_path}: {e}")
        return False

# VGG16 for Deep Features (GPU)
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
vgg_model = tf.keras.Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block5_pool').output)

def extract_vgg_features_batch(images):
    img_arrays = np.array([img_to_array(cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), (48, 48))) for img in images])
    features = vgg_model.predict(img_arrays, verbose=0, batch_size=32)
    return features.reshape(len(images), -1)

# Augmentation Pipeline
aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7)
])

# Load Images
IMG_SIZE = (48, 48)
def load_images(folder, label, batch_size=1000, max_images=5000):
    data = []
    filenames = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(filenames)
    filenames = filenames[:max_images]  # Limit to 5000 images
    batch_images = []
    for i in tqdm(range(0, len(filenames), batch_size), desc=f"Loading {folder}"):
        batch_files = filenames[i:i+batch_size]
        for filename in batch_files:
            img_path = os.path.join(folder, filename)
            if not is_valid_face(img_path):
                continue
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                # Original
                batch_images.append(img)
                if len(batch_images) >= batch_size:
                    vgg_features = extract_vgg_features_batch(batch_images)
                    for j, features in enumerate(vgg_features):
                        img_pil = Image.fromarray(batch_images[j]).resize(IMG_SIZE)
                        arr = np.concatenate([np.array(img_pil).flatten(), features])
                        data.append((arr, label))
                    batch_images = []
                # Augmentation (1 per image)
                aug_img = aug(image=img)['image']
                batch_images.append(aug_img)
                if len(batch_images) >= batch_size:
                    vgg_features = extract_vgg_features_batch(batch_images)
                    for j, features in enumerate(vgg_features):
                        aug_pil = Image.fromarray(batch_images[j]).resize(IMG_SIZE)
                        arr = np.concatenate([np.array(aug_pil).flatten(), features])
                        data.append((arr, label))
                    batch_images = []
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                continue
        log_memory()
    # Process remaining images
    if batch_images:
        vgg_features = extract_vgg_features_batch(batch_images)
        for j, features in enumerate(vgg_features):
            img_pil = Image.fromarray(batch_images[j]).resize(IMG_SIZE)
            arr = np.concatenate([np.array(img_pil).flatten(), features])
            data.append((arr, label))
    return data

# Load + Balance Dataset
start_time = time.time()
man_data = load_images(man_dir, label=0)
woman_data = load_images(woman_dir, label=1)

min_samples = min(len(man_data), len(woman_data))
man_data = man_data[:min_samples]
woman_data = woman_data[:min_samples]
logger.info(f"After balancing - Men: {len(man_data)}, Women: {len(woman_data)}")

data = man_data + woman_data
random.shuffle(data)

X = np.array([x[0] for x in data])
y = np.array([x[1] for x in data])
logger.info(f"Final dataset - Men: {sum(y == 0)}, Women: {sum(y == 1)}")
logger.info(f"Data loading took {time.time() - start_time:.2f} seconds")

# Normalize & Split
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Log dataset sizes
total_samples = len(X_scaled)
train_samples = len(X_train)
test_samples = len(X_test)
test_percentage = (test_samples / total_samples) * 100
logger.info(f"Dataset split - Total: {total_samples}, Train: {train_samples}, Test: {test_samples}")
logger.info(f"Test dataset percentage: {test_percentage:.2f}%")

# Custom CuPy PCA
class CuPyPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None

    def fit_transform(self, X):
        X_cp = cp.array(X)
        self.mean_ = cp.mean(X_cp, axis=0)
        X_centered = X_cp - self.mean_
        cov = cp.cov(X_centered.T)
        eigenvalues, eigenvectors = cp.linalg.eigh(cov)
        idx = cp.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        total_variance = cp.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / total_variance
        self.components_ = eigenvectors[:, :self.n_components]
        return cp.asnumpy(X_centered @ self.components_)

    def transform(self, X):
        X_cp = cp.array(X)
        X_centered = X_cp - self.mean_
        return cp.asnumpy(X_centered @ self.components_)

# Feature Selection
feature_selectors = {
    "PCA": CuPyPCA(n_components=50),
    "VarianceThreshold": VarianceThreshold(threshold=0.005),
    "SelectKBest": SelectKBest(mutual_info_classif, k=200)
}

# Classifiers
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, class_weight='balanced'),
    "RandomForest": RandomForestClassifier(class_weight='balanced'),
    "XGBoost": XGBClassifier(device='cuda', eval_metric='logloss', early_stopping_rounds=10)
}

# Hyperparameter Grids (Reduced)
param_grids = {
    "LogisticRegression": {'C': [1, 10], 'solver': ['lbfgs'], 'penalty': ['l2']},
    "KNN": {'n_neighbors': [5, 7], 'weights': ['distance']},
    "SVM": {'C': [1], 'kernel': ['rbf'], 'gamma': ['scale']},
    "RandomForest": {'n_estimators': [100], 'max_depth': [None], 'min_samples_split': [2]},
    "XGBoost": {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [6]}
}

# Evaluation Loop
best_score = 0
best_model = None
best_selector = None
final_pca = None
best_model_name = ""
best_selector_name = ""
all_results = []
trained_models = []

for selector_name, selector in feature_selectors.items():
    logger.info(f"\n--- Feature Selection: {selector_name} ---")
    start_time = time.time()
    
    if selector_name == "PCA":
        X_train_sel = selector.fit_transform(X_train)
        X_test_sel = selector.transform(X_test)
        current_pca = selector
        logger.info(f"PCA components: {selector.n_components}, Explained variance: {sum(selector.explained_variance_ratio_):.4f}")
    else:
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)
        current_pca = None
        logger.info(f"Features after {selector_name}: {X_train_sel.shape[1]}")

    logger.info(f"Feature selection took {time.time() - start_time:.2f} seconds")

    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name} with {selector_name}...")
        start_time = time.time()
        
        grid = GridSearchCV(model, param_grids[model_name], cv=3, scoring='accuracy', n_jobs=-1)
        if model_name == "XGBoost":
            weights = compute_sample_weight(class_weight='balanced', y=y_train)
            grid.fit(X_train_sel, y_train, sample_weight=weights, eval_set=[(X_test_sel, y_test)], verbose=False)
        else:
            grid.fit(X_train_sel, y_train)
        
        model = grid.best_estimator_
        logger.info(f"Best parameters: {grid.best_params_}")

        y_pred = model.predict(X_test_sel)
        acc = model.score(X_test_sel, y_test)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, model.predict_proba(X_test_sel)[:, 1]) if hasattr(model, "predict_proba") else 0.0

        logger.info(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f} | AUC: {auc:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred, target_names=["Man", "Woman"]))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Man", "Woman"], yticklabels=["Man", "Woman"])
        plt.title(f"{model_name} - {selector_name} Confusion Matrix")
        plt.tight_layout()
        cm_path = os.path.join(output_dir, f"{model_name}_{selector_name}_cm.jpg")
        plt.savefig(cm_path)
        plt.close()
        display(IPImage(filename=cm_path))

        all_results.append((selector_name, model_name, acc))
        trained_models.append((model_name, selector_name, model, acc))

        if acc > best_score:
            best_score = acc
            best_model = model
            best_model_name = model_name
            best_selector = selector
            best_selector_name = selector_name
            final_pca = current_pca

        logger.info(f"Model evaluation took {time.time() - start_time:.2f} seconds")

# Ensemble Stacking
logger.info("\n--- Training Ensemble ---")
top_models = sorted(trained_models, key=lambda x: x[3], reverse=True)[:3]
# Clone models and disable early stopping for XGBoost in ensemble
ensemble_models = []
for name, model in [(f"{m[0]}_{m[1]}", m[2]) for m in top_models]:
    if isinstance(model, XGBClassifier) and model.early_stopping_rounds is not None:
        # Clone the model and set early_stopping_rounds to None
        from copy import deepcopy
        model_clone = deepcopy(model)
        model_clone.early_stopping_rounds = None
        ensemble_models.append((name, model_clone))
    else:
        ensemble_models.append((name, model))

ensemble = StackingClassifier(estimators=ensemble_models, final_estimator=LogisticRegression(), n_jobs=-1)
ensemble.fit(X_train_sel, y_train)
y_pred_ensemble = ensemble.predict(X_test_sel)
acc_ensemble = ensemble.score(X_test_sel, y_test)
logger.info(f"Ensemble Accuracy: {acc_ensemble:.4f}")
logger.info("\n" + classification_report(y_test, y_pred_ensemble, target_names=["Man", "Woman"]))

# Ensemble Confusion Matrix
cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_ensemble, annot=True, fmt="d", cmap="Blues", xticklabels=["Man", "Woman"], yticklabels=["Man", "Woman"])
plt.title("Ensemble Classifier Confusion Matrix")
plt.tight_layout()
cm_ensemble_path = os.path.join(output_dir, "ensemble_cm.jpg")
plt.savefig(cm_ensemble_path)
plt.close()
display(IPImage(filename=cm_ensemble_path))

# Update best model
if acc_ensemble > best_score:
    best_score = acc_ensemble
    best_model = ensemble
    best_model_name = "Ensemble"
    best_selector_name = selector_name

# Model Comparison Plot
plt.figure(figsize=(10, 6))
for sel in feature_selectors:
    sel_scores = [acc for s, m, acc in all_results if s == sel]
    sel_labels = [m for s, m, acc in all_results if s == sel]
    plt.plot(sel_labels, sel_scores, marker='o', label=sel)
plt.plot(['Ensemble'], [acc_ensemble], marker='*', markersize=15, label='Ensemble')
plt.title("Model Accuracy by Feature Selection")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
comp_path = os.path.join(output_dir, "model_comparison.jpg")
plt.savefig(comp_path)
plt.close()
display(IPImage(filename=comp_path))

# Save Artifacts
logger.info(f"\nBest Model: {best_model_name} ({best_selector_name}) with Accuracy: {best_score:.4f}")
joblib.dump(best_model, os.path.join(output_dir, "model.pkl"), compress=3)
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"), compress=3)
joblib.dump(final_pca, os.path.join(output_dir, "pca.pkl"), compress=3) if final_pca else joblib.dump(None, os.path.join(output_dir, "pca.pkl"), compress=3)
joblib.dump(best_selector, os.path.join(output_dir, "selector.pkl"), compress=3)

logger.info("Training complete and files saved to Google Drive.")