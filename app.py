"""
Aplikasi Streamlit: Clustering Gaya Hidup & Performa Akademik Mahasiswa (K-Means).

User memasukkan gaya hidupnya lewat slider di sidebar, lalu aplikasi memprediksi
mahasiswa tersebut masuk ke cluster yang mana, menampilkan rata-rata tiap cluster,
dan memvisualisasikan posisinya pada peta cluster (PCA 2D).
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------------
# Konfigurasi & konstanta
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Clustering Gaya Hidup Mahasiswa",
    page_icon="🎓",
    layout="wide",
)

STRESS_MAPPING = {"Low": 0, "Moderate": 1, "High": 2}

# Urutan fitur dijaga konsisten antara data latih dan input user
# (penting supaya scaler & K-Means tidak salah memetakan kolom).
FEATURES = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "GPA",
    "Stress_Level",
]

N_CLUSTERS = 2


# ----------------------------------------------------------------------------
# Load data + latih model (di-cache supaya tidak dilatih ulang tiap slider digeser)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_and_train():
    data = pd.read_csv("student_lifestyle_dataset.csv")
    data["Stress_Level"] = data["Stress_Level"].map(STRESS_MAPPING).astype(int)
    data = data.drop(columns=["Student_ID"])
    data = data[FEATURES]  # pastikan urutan kolom konsisten

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)

    pca = PCA(n_components=2, random_state=42)
    data_pca = pca.fit_transform(data_scaled)

    data = data.copy()
    data["Cluster"] = labels
    return data, scaler, kmeans, pca, data_pca


data, scaler, kmeans, pca, data_pca = load_and_train()

# Ringkasan rata-rata tiap cluster + deskripsi otomatis.
# Deskripsi diturunkan dari data (bukan hardcode) supaya tetap benar meski
# penomoran cluster berbeda antar versi scikit-learn.
cluster_summary = data.groupby("Cluster")[FEATURES].mean()
high_gpa_cluster = cluster_summary["GPA"].idxmax()

cluster_desc = {}
for c in cluster_summary.index:
    if c == high_gpa_cluster:
        cluster_desc[c] = (
            "Fokus belajar tinggi & GPA lebih baik, tapi tingkat stres cenderung "
            "lebih tinggi."
        )
    else:
        cluster_desc[c] = (
            "Lebih banyak aktivitas fisik & sosial, jam belajar moderat, GPA lebih "
            "rendah, dengan tingkat stres yang lebih rendah."
        )


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🎓 Prediksi Cluster Gaya Hidup & Performa Akademik Mahasiswa")
st.write(
    "Aplikasi ini mengelompokkan mahasiswa berdasarkan gaya hidup, performa "
    "akademik, dan tingkat stres menggunakan **K-Means Clustering**. "
    "Atur gaya hidupmu di sidebar kiri untuk melihat kamu masuk cluster yang mana."
)


# ----------------------------------------------------------------------------
# Sidebar: input fitur
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Input Gaya Hidupmu")
study_hours = st.sidebar.slider("Jam Belajar / Hari", 0.1, 12.0, 5.0)
extracurricular_hours = st.sidebar.slider("Jam Ekstrakurikuler / Hari", 0.0, 10.0, 2.0)
sleep_hours = st.sidebar.slider("Jam Tidur / Hari", 4.0, 10.0, 7.0)
social_hours = st.sidebar.slider("Jam Bersosialisasi / Hari", 0.0, 12.0, 3.0)
physical_activity = st.sidebar.slider("Jam Aktivitas Fisik / Hari", 0.0, 10.0, 2.0)
gpa = st.sidebar.slider("GPA", 0.0, 4.0, 3.0)
stress_level = st.sidebar.selectbox("Tingkat Stres", list(STRESS_MAPPING.keys()))

user_input = pd.DataFrame(
    [[
        study_hours,
        extracurricular_hours,
        sleep_hours,
        social_hours,
        physical_activity,
        gpa,
        STRESS_MAPPING[stress_level],
    ]],
    columns=FEATURES,
)


# ----------------------------------------------------------------------------
# Prediksi cluster untuk input user
# ----------------------------------------------------------------------------
user_scaled = scaler.transform(user_input)
user_cluster = int(kmeans.predict(user_scaled)[0])

st.subheader("📝 Data Input Kamu")
st.dataframe(user_input, use_container_width=True)

st.subheader("🎯 Hasil Prediksi")
st.success(f"Kamu diprediksi masuk ke **Cluster {user_cluster}**")
st.write(cluster_desc[user_cluster])

# ----------------------------------------------------------------------------
# Rata-rata fitur per cluster
# ----------------------------------------------------------------------------
st.subheader("📊 Rata-Rata Fitur per Cluster")
st.dataframe(
    cluster_summary.style.format("{:.2f}").highlight_max(axis=0, color="#2e7d32"),
    use_container_width=True,
)

# ----------------------------------------------------------------------------
# Visualisasi cluster (PCA 2D) + posisi input user
# ----------------------------------------------------------------------------
st.subheader("🗺️ Peta Cluster (PCA 2D)")

# Proyeksikan input user ke ruang PCA yang SAMA dengan data latih,
# supaya titik merahnya jatuh di posisi yang benar.
user_pca = pca.transform(user_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
for cluster in sorted(data["Cluster"].unique()):
    mask = data["Cluster"] == cluster
    ax.scatter(
        data_pca[mask.values, 0],
        data_pca[mask.values, 1],
        label=f"Cluster {cluster}",
        alpha=0.5,
        s=40,
    )
ax.scatter(
    user_pca[0, 0],
    user_pca[0, 1],
    c="red",
    s=280,
    marker="X",
    edgecolors="black",
    linewidths=1.5,
    label="Input Kamu",
    zorder=5,
)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("Visualisasi Clustering K-Means")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

st.caption(
    "Dibuat dengan Streamlit • Dataset: Student Lifestyle (2000 mahasiswa) • "
    "Metode: StandardScaler + K-Means (k=2) + PCA untuk visualisasi."
)
