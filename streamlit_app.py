import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Model bileşenlerini yükle
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("features.pkl")  # PCA bu isimle kaydedildi

st.set_page_config(page_title="Yüz Görüntüsünden Cinsiyet Tanıma", layout="centered")

st.title("Yüz Görüntüsünden Cinsiyet Tanıma")
st.write("Bir yüz resmi yükleyin (gri tonlama veya renkli), model cinsiyeti tahmin edecektir.")

uploaded_file = st.file_uploader("Bir resim seçin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Adım 1: Görüntüyü oku ve ön işle
        image = Image.open(uploaded_file).convert("L")
        image_resized = image.resize((32, 32))  # Eğitim boyutuyla uyumlu olmalı
        img_array = np.array(image_resized).flatten().reshape(1, -1)

        st.image(image_resized, caption="İşlenmiş Görüntü", width=150)
        st.write("Görüntü yüklendi ve ön işlemden geçirildi.")

        # Adım 2: Ölçeklendir + PCA
        img_scaled = scaler.transform(img_array)
        img_pca = pca.transform(img_scaled)

        # Adım 3: Tahmin yap
        prediction = model.predict(img_pca)[0]
        proba = model.predict_proba(img_pca)[0]

        label = "Erkek" if prediction == 0 else "Kadın"
        st.success(f"**Tahmin:** {label}")
        st.write(f"Güven: Erkek %{proba[0]*100:.2f}, Kadın %{proba[1]*100:.2f}")

    except Exception as e:
        st.error(f"Görüntü işlenirken hata oluştu: {e}")
