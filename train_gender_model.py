import os
import numpy as np
from PIL import Image
import kagglehub
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 1: Download dataset
path = kagglehub.dataset_download("maciejgronczynski/biggest-genderface-recognition-dataset")
print("Dataset path:", path)

def load_images_from_folder(folder, label, img_size=(128,128)):
    data = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert("L")
            img = img.resize(img_size)
            img_array = np.array(img).flatten()
            data.append((img_array, label))
        except:
            continue
    return data

# Step 2: Load data
man_dir = os.path.join(path, "faces", "man")
woman_dir = os.path.join(path, "faces", "woman")
man_data = load_images_from_folder(man_dir, label=0)
woman_data = load_images_from_folder(woman_dir, label=1)

print(f"Before balancing - Men: {len(man_data)}, Women: {len(woman_data)}")

# Balance dataset by oversampling minority class
if len(man_data) > len(woman_data):
    diff = len(man_data) - len(woman_data)
    additional_women = random.choices(woman_data, k=diff)
    woman_data += additional_women
elif len(woman_data) > len(man_data):
    diff = len(woman_data) - len(man_data)
    additional_men = random.choices(man_data, k=diff)
    man_data += additional_men

all_data = man_data + woman_data
random.shuffle(all_data)

X = np.array([i[0] for i in all_data])
y = np.array([i[1] for i in all_data])

print(f"After balancing - Men: {sum(y==0)}, Women: {sum(y==1)}")

# Step 3: Normalize
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42
)

# Step 5: PCA
pca = PCA(n_components=1000, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Step 6: VarianceThreshold
vt = VarianceThreshold(threshold=0.01)
X_train_vt = vt.fit_transform(X_train_pca)
X_test_vt = vt.transform(X_test_pca)

# Step 7: SelectKBest
selector = SelectKBest(mutual_info_classif, k=300)
X_train_selected = selector.fit_transform(X_train_vt, y_train)
X_test_selected = selector.transform(X_test_vt)

# Step 8: Evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, name=""):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Results for {name}")
    print(classification_report(y_test, y_pred, target_names=["Man", "Woman"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Man", "Woman"], yticklabels=["Man", "Woman"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()

print("\n--- Using PCA + VarianceThreshold + SelectKBest Features ---")
evaluate_model(LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced'), 
               X_train_selected, X_test_selected, y_train, y_test, "Logistic Regression (Balanced)")

evaluate_model(DecisionTreeClassifier(random_state=42), 
               X_train_selected, X_test_selected, y_train, y_test, "Decision Tree")

evaluate_model(KNeighborsClassifier(n_neighbors=5), 
               X_train_selected, X_test_selected, y_train, y_test, "KNN")

evaluate_model(SVC(probability=True, class_weight='balanced', random_state=42), 
               X_train_selected, X_test_selected, y_train, y_test, "SVM (Balanced)")

# Step 9: Save final model with compression
joblib.dump(SVC(probability=True, class_weight='balanced', random_state=42).fit(X_train_selected, y_train), "model.pkl", compress=3)
joblib.dump(scaler, "scaler.pkl", compress=3)
joblib.dump(pca, "pca.pkl", compress=3)
joblib.dump(vt, "variance_threshold.pkl", compress=3)
joblib.dump(selector, "selector.pkl", compress=3)

print("\nFinal model, scaler, PCA, VarianceThreshold, and selector saved successfully with compression!")