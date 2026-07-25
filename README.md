# 🎓 Student Lifestyle Clustering (K-Means)

Aplikasi web interaktif yang mengelompokkan mahasiswa berdasarkan **gaya hidup, performa akademik, dan tingkat stres** menggunakan **K-Means Clustering**. User cukup mengatur gaya hidupnya lewat slider, lalu aplikasi memprediksi ia masuk ke cluster mana beserta visualisasinya.

> 🔗 **Live demo:** https://student-lifestyle-clustering.streamlit.app

<!-- Tips: tambahkan screenshot aplikasi di sini setelah deploy -->
<!-- ![Screenshot](screenshot.png) -->

---

## ✨ Fitur
- Input gaya hidup interaktif (jam belajar, tidur, sosial, aktivitas fisik, GPA, stres).
- Prediksi cluster secara real-time memakai model K-Means terlatih.
- Tabel rata-rata tiap cluster + interpretasi otomatis.
- Visualisasi peta cluster (reduksi dimensi **PCA 2D**) lengkap dengan posisi input user.

## 📊 Dataset
`student_lifestyle_dataset.csv` — 2000 mahasiswa, 7 fitur:

| Fitur | Keterangan |
|---|---|
| `Study_Hours_Per_Day` | Jam belajar per hari |
| `Extracurricular_Hours_Per_Day` | Jam ekstrakurikuler per hari |
| `Sleep_Hours_Per_Day` | Jam tidur per hari |
| `Social_Hours_Per_Day` | Jam bersosialisasi per hari |
| `Physical_Activity_Hours_Per_Day` | Jam aktivitas fisik per hari |
| `GPA` | Indeks prestasi (0–4) |
| `Stress_Level` | Tingkat stres (Low / Moderate / High) |

## 🧠 Metode
1. **Preprocessing** — encode `Stress_Level` ke numerik, buang kolom `Student_ID`.
2. **Normalisasi** — `StandardScaler` supaya semua fitur setara skalanya.
3. **Clustering** — `K-Means` (k = 2, dipilih lewat *Elbow Method* & *Silhouette Score* di notebook).
4. **Visualisasi** — `PCA` menurunkan 7 dimensi jadi 2 supaya cluster bisa dipetakan.

### Interpretasi cluster
- **Cluster fokus akademik** — jam belajar tinggi, GPA lebih baik, tapi stres cenderung lebih tinggi.
- **Cluster seimbang/aktif** — lebih banyak aktivitas fisik & sosial, jam belajar moderat, GPA lebih rendah, stres lebih rendah.

## 🛠️ Tech Stack
Python · Streamlit · scikit-learn · pandas · NumPy · Matplotlib

## 🚀 Menjalankan secara lokal
```bash
git clone https://github.com/byochiram/student-lifestyle-clustering.git
cd student-lifestyle-clustering
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Struktur
```
student-lifestyle-clustering/
├── app.py                          # Aplikasi Streamlit
├── student_lifestyle_dataset.csv   # Dataset
├── requirements.txt                # Dependensi
└── README.md
```

## 👤 Tentang
Berawal dari project Machine Learning, lalu dikembangkan menjadi aplikasi web interaktif untuk portfolio.
