import os
import numpy as np
from PIL import Image
import kagglehub
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
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
print("✅ Dataset path:", path)

# Step 2: Load images
def load_images_from_folder(folder, label, img_size=(32, 32)):
    data = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert("L")  # Grayscale
            img = img.resize(img_size)
            img_array = np.array(img).flatten()
            data.append((img_array, label))
        except:
            continue
    return data

# Step 3: Prepare data
man_dir = os.path.join(path, "faces", "man")
woman_dir = os.path.join(path, "faces", "woman")
man_data = load_images_from_folder(man_dir, label=0)
woman_data = load_images_from_folder(woman_dir, label=1)

all_data = man_data + woman_data
np.random.shuffle(all_data)

X = np.array([i[0] for i in all_data])
y = np.array([i[1] for i in all_data])

print("✅ Total samples:", len(X))
print("✅ Image size (flattened):", X.shape[1])
print("✅ Labels shape:", y.shape)

# Step 4: Normalize features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

print("✅ Normalized shape:", X_normalized.shape)
print("Min value:", X_normalized.min(), "| Max value:", X_normalized.max())

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Step 6: PCA
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print("✅ PCA shape:", X_train_pca.shape)

# Step 7: Feature Selection (Optional: Uncomment if needed)
# SelectKBest
skb = SelectKBest(score_func=f_classif, k=100)
X_train_skb = skb.fit_transform(X_train, y_train)
X_test_skb = skb.transform(X_test)

# RFE
rfe_model = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=rfe_model, n_features_to_select=100, step=50)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Step 8: Model evaluation
def evaluate_model(model, X_train, X_test, y_train, y_test, name=""):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n📊 Results for {name}")
    print(classification_report(y_test, y_pred, target_names=["Man", "Woman"]))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Man", "Woman"], yticklabels=["Man", "Woman"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Step 9: Try all models with PCA features
print("\n--- Using PCA Features ---")
evaluate_model(LogisticRegression(max_iter=1000), X_train_pca, X_test_pca, y_train, y_test, "Logistic Regression (PCA)")
evaluate_model(DecisionTreeClassifier(), X_train_pca, X_test_pca, y_train, y_test, "Decision Tree (PCA)")
evaluate_model(KNeighborsClassifier(n_neighbors=5), X_train_pca, X_test_pca, y_train, y_test, "KNN (PCA)")
evaluate_model(SVC(probability=True), X_train_pca, X_test_pca, y_train, y_test, "SVM (PCA)")

# Step 10: Save final model with PCA
final_model = SVC(probability=True)
final_model.fit(X_train_pca, y_train)

joblib.dump(final_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "features.pkl")  # Keep name consistent with Streamlit app

print("\n✅ Final model, scaler, and feature selector saved successfully!")