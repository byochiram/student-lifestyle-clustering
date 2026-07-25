"""
Aplikasi Streamlit: Pengelompokan Gaya Hidup Mahasiswa (K-Means Clustering).

Mengelompokkan mahasiswa menjadi tipe-tipe gaya hidup berdasarkan kebiasaan sehari-hari
(belajar, tidur, olahraga, sosial) dan tingkat stres — tanpa label yang sudah ada
sebelumnya (unsupervised). User bisa mengecek satu mahasiswa lewat slider, atau
mengelompokkan banyak mahasiswa sekaligus lewat upload CSV.
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
st.set_page_config(page_title="Tipe Gaya Hidup Mahasiswa", page_icon="🎓", layout="wide")

STRESS_MAPPING = {"Low": 0, "Moderate": 1, "High": 2}

FEATURES = [
    "Study_Hours_Per_Day", "Extracurricular_Hours_Per_Day", "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day", "Physical_Activity_Hours_Per_Day", "GPA", "Stress_Level",
]

# Nama kolom yang ramah dibaca.
COL_LABELS = {
    "Study_Hours_Per_Day": "Jam Belajar",
    "Extracurricular_Hours_Per_Day": "Jam Ekskul",
    "Sleep_Hours_Per_Day": "Jam Tidur",
    "Social_Hours_Per_Day": "Jam Sosial",
    "Physical_Activity_Hours_Per_Day": "Jam Olahraga",
    "GPA": "GPA",
    "Stress_Level": "Stres (0–2)",
}

N_CLUSTERS = 2


# ----------------------------------------------------------------------------
# Load data + latih model (di-cache supaya tidak dilatih ulang tiap slider digeser)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_and_train():
    data = pd.read_csv("student_lifestyle_dataset.csv")
    data["Stress_Level"] = data["Stress_Level"].map(STRESS_MAPPING).astype(int)
    data = data.drop(columns=["Student_ID"])[FEATURES]

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
medians = data[FEATURES].median()

# Ringkasan + nama kelompok yang manusiawi (diturunkan dari data, bukan hardcode,
# supaya tetap benar meski penomoran cluster berbeda antar versi scikit-learn).
cluster_summary = data.groupby("Cluster")[FEATURES].mean()
high_gpa_cluster = cluster_summary["GPA"].idxmax()

NAMES, DESCS = {}, {}
for c in cluster_summary.index:
    if c == high_gpa_cluster:
        NAMES[c] = "🎯 Si Fokus Akademik"
        DESCS[c] = "Jam belajar tinggi & GPA lebih bagus, tapi tingkat stres cenderung lebih tinggi."
    else:
        NAMES[c] = "🌿 Si Aktif & Seimbang"
        DESCS[c] = ("Lebih banyak aktivitas fisik & sosial, jam belajar moderat, "
                    "GPA sedikit lebih rendah, dan tingkat stres lebih rendah.")


def prepare(frame):
    """Siapkan data mentah (satu/banyak baris) agar bisa diprediksi K-Means."""
    out = frame.copy()
    if "Student_ID" in out.columns:
        out = out.drop(columns=["Student_ID"])
    # Stress_Level boleh berupa teks (Low/Moderate/High) atau angka.
    if out["Stress_Level"].dtype == object:
        out["Stress_Level"] = out["Stress_Level"].map(STRESS_MAPPING)
    out = out[FEATURES]
    for col in FEATURES:  # amankan tipe + isi missing dengan median data latih
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(medians[col])
    return out


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🎓 Tipe Gaya Hidup Mahasiswa")
st.write(
    "Aplikasi ini mengelompokkan mahasiswa menjadi **2 tipe gaya hidup** berdasarkan "
    "kebiasaan sehari-hari (belajar, tidur, olahraga, sosialisasi) dan tingkat stres. "
    "Atur gaya hidupmu di sidebar kiri untuk melihat kamu tipe yang mana."
)
st.caption(
    "Metode: **K-Means Clustering** — komputer menemukan sendiri kelompok mahasiswa yang "
    "mirip, tanpa diberi jawaban sebelumnya (unsupervised)."
)


# ----------------------------------------------------------------------------
# Sidebar: input satu mahasiswa
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Gaya Hidupmu")
study_hours = st.sidebar.slider("Jam Belajar / Hari", 0.1, 12.0, 5.0)
extracurricular_hours = st.sidebar.slider("Jam Ekstrakurikuler / Hari", 0.0, 10.0, 2.0)
sleep_hours = st.sidebar.slider("Jam Tidur / Hari", 4.0, 10.0, 7.0)
social_hours = st.sidebar.slider("Jam Bersosialisasi / Hari", 0.0, 12.0, 3.0)
physical_activity = st.sidebar.slider("Jam Aktivitas Fisik / Hari", 0.0, 10.0, 2.0)
gpa = st.sidebar.slider("GPA", 0.0, 4.0, 3.0)
stress_level = st.sidebar.radio("Tingkat Stres", list(STRESS_MAPPING.keys()))

user_input = pd.DataFrame([[
    study_hours, extracurricular_hours, sleep_hours, social_hours,
    physical_activity, gpa, STRESS_MAPPING[stress_level],
]], columns=FEATURES)


# ----------------------------------------------------------------------------
# Prediksi satu mahasiswa
# ----------------------------------------------------------------------------
user_scaled = scaler.transform(user_input)
user_cluster = int(kmeans.predict(user_scaled)[0])

st.subheader("🎯 Hasil")
st.success(f"Kamu paling mirip dengan kelompok **{NAMES[user_cluster]}**")
st.write(DESCS[user_cluster])

with st.expander("Lihat rincian data yang kamu isi"):
    show = user_input.rename(columns=COL_LABELS)
    st.dataframe(show, use_container_width=True)


# ----------------------------------------------------------------------------
# Karakter tiap kelompok
# ----------------------------------------------------------------------------
st.subheader("📊 Karakter Tiap Kelompok")
st.write("Rata-rata kebiasaan tiap kelompok — biar kebayang tiap tipe itu seperti apa.")
friendly = cluster_summary.rename(index=NAMES, columns=COL_LABELS)
st.dataframe(
    friendly.style.format("{:.2f}").highlight_max(axis=0, color="#1b5e20"),
    use_container_width=True,
)
st.caption(f"Baris = kelompok. Kamu diprediksi masuk **{NAMES[user_cluster]}**.")


# ----------------------------------------------------------------------------
# Peta kelompok (visualisasi PCA 2D) + posisi kamu
# ----------------------------------------------------------------------------
st.subheader("🗺️ Peta Kelompok")
st.write(
    "Tiap titik = 1 mahasiswa; warna = kelompoknya. Tanda **X merah** = posisi kamu. "
    "(7 kebiasaan diringkas jadi 2 sumbu pakai PCA supaya bisa digambar.)"
)
user_pca = pca.transform(user_scaled)
fig, ax = plt.subplots(figsize=(8, 6))
for c in sorted(data["Cluster"].unique()):
    m = (data["Cluster"] == c).values
    ax.scatter(data_pca[m, 0], data_pca[m, 1], label=NAMES[c], alpha=0.5, s=40)
ax.scatter(user_pca[0, 0], user_pca[0, 1], c="red", s=280, marker="X",
           edgecolors="black", linewidths=1.5, label="Kamu", zorder=5)
ax.set_xlabel("Komponen 1")
ax.set_ylabel("Komponen 2")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)


# ----------------------------------------------------------------------------
# Kelompokkan banyak mahasiswa (data contoh ATAU upload sendiri)
# ----------------------------------------------------------------------------
st.subheader("📦 Kelompokkan Banyak Mahasiswa Sekaligus")
st.write(
    "Punya data banyak mahasiswa? Upload file CSV-nya untuk mengelompokkan semuanya "
    "sekaligus — atau coba dulu dengan data contoh. Tiap mahasiswa baru akan dimasukkan "
    "ke kelompok terdekat dari 2 kelompok yang sudah terbentuk."
)
mode = st.radio(
    "Sumber data:",
    ["🎲 Pakai data contoh (2000 mahasiswa)", "📤 Upload file CSV saya sendiri"],
)

source = None
if mode.startswith("🎲"):
    source = pd.read_csv("student_lifestyle_dataset.csv")
else:
    st.caption(
        "CSV harus punya kolom: " + ", ".join(FEATURES) +
        " · Stress_Level boleh Low/Moderate/High atau 0/1/2 · Student_ID opsional."
    )
    up = st.file_uploader("Pilih file CSV", type=["csv"])
    if up is not None:
        try:
            source = pd.read_csv(up)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")

if source is not None:
    missing = [c for c in FEATURES if c not in source.columns]
    if missing:
        st.error("File-nya kurang kolom: " + ", ".join(missing))
    else:
        ids = source["Student_ID"] if "Student_ID" in source.columns else range(1, len(source) + 1)
        labels = kmeans.predict(scaler.transform(prepare(source)))
        res = pd.DataFrame({"Student_ID": ids, "Cluster": labels})
        res["Kelompok"] = res["Cluster"].map(NAMES)
        st.write(f"✅ **{len(res)} mahasiswa** dikelompokkan. Sebarannya:")
        st.bar_chart(res["Kelompok"].value_counts())
        st.dataframe(res.head(20), use_container_width=True)
        st.download_button(
            "⬇️ Download hasil lengkap (CSV)",
            res.to_csv(index=False).encode("utf-8"),
            file_name="hasil_kelompok_mahasiswa.csv",
            mime="text/csv",
        )

st.caption(
    "Dibuat dengan Streamlit • Dataset: Student Lifestyle (2000 mahasiswa) • "
    "Metode: StandardScaler + K-Means (k=2) + PCA untuk visualisasi."
)
