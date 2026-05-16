# Image Captioning with CNN & RNN
**Tubes2ML-17-k01** - Implementation of CNN and RNN from Scratch for Image Captioning

Proyek ini mengimplementasikan sistem image captioning end-to-end yang menggabungkan Convolutional Neural Networks (CNN) untuk ekstraksi fitur visual dan Recurrent Neural Networks (RNN/LSTM) untuk generasi teks deskriptif. Seluruh komponen dibangun dari scratch (without external libraries) untuk pembelajaran mendalam tentang deep learning fundamentals, kemudian divalidasi dengan model Keras equivalents.

**Dataset**: Intel Image Dataset (CNN) + Flickr8k (Image Captioning)  
**Metrics**: Macro F1 Score, BLEU-4, METEOR  
**Architecture**: CNN Encoder (from scratch + Keras) → LSTM Decoder (from scratch + Keras)

---

## Daftar Isi
- [Repository Overview](#repository-overview)
- [Setup Instructions](#setup-instructions)
- [Running the Program](#running-the-program)
- [Project Structure](#project-structure)
- [Task Division](#pembagian-tugas-kelompok)

---

## Repository Overview

Repositori ini berisi implementasi lengkap dari:

1. **CNN from Scratch**: Conv2D, ReLU, MaxPooling, Flatten layers tanpa NumPy-only backend
2. **RNN/LSTM from Scratch**: SimpleRNNCell, LSTMCell (dengan input, forget, cell, output gates)
3. **Embedding Layer**: Token-to-vector transformation dari scratch
4. **Keras Models**: Equivalent architectures untuk validation dan comparison
5. **Image Captioning Pipeline**: End-to-end inference dengan greedy decoding dan beam search
6. **Evaluation Metrics**: BLEU-4, METEOR, qualitative analysis

---

## Setup Instructions

### **Prerequisite**
- Python 3.8+
- pip atau conda

### **1. Clone Repository**
```bash
git clone https://github.com/wrdtlkhoir/Tubes2ML-17-k01.git
cd Tubes2ML-17-k01
```

### **2. Create Virtual Environment** (Recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**Key dependencies**:
- NumPy: Numerical computations
- TensorFlow/Keras: Model building and training
- Matplotlib, Seaborn: Visualizations
- NLTK: Text preprocessing
- OpenCV: Image processing

---

## Running the Program

### **1. Train CNN Models**
```bash
python src/main_train.py
```
Ini akan melatih berbagai variasi CNN pada Intel Image Dataset dan menyimpan weights.

### **2. Extract Image Features (Image Captioning)**
```bash
cd src/rnn
python preprocessing.py
```
Menggunakan pretrained VGG16/InceptionV3 untuk ekstraksi features dari Flickr8k images.

### **3. Train RNN/LSTM Models**
```bash
python cnn/train.py  # CNN models
python rnn/train.py  # RNN/LSTM models
```
Melatih 12+ variasi decoder LSTM untuk caption generation.

### **4. Run Complete Pipeline**
```bash
# Inference dengan from-scratch implementation
python src/pipeline/run_inference.py

# Atau jalankan notebook untuk interactive exploration
jupyter notebook runner.ipynb
```

### **5. Evaluate Results**
```bash
python src/evaluation/metrics.py
```
Menghitung BLEU-4, METEOR scores dan generate qualitative analysis.

### **6. Generate Bonus Visualizations**
```bash
cd src/cnn
jupyter notebook bonus_cnn.ipynb
```
Menjalankan feature map visualization, Grad-CAM, gradient checking, dll.

---

## Project Structure

```
Tubes2ML-17-k01/
├── README.md                          # Dokumentasi utama
├── requirements.txt                   # Dependencies
├── environment.yml                    # Conda environment
│
├── data/                              # Dataset directory
│   ├── intel_images/                  # Intel Image Classification
│   └── flickr8k/                      # Flickr8k Image Captioning
│
├── src/
│   ├── main_train.py                  # Entry point untuk training
│   │
│   ├── cnn/                           # CNN implementations
│   │   ├── layers.py                  # Layer definitions
│   │   ├── models.py                  # Model architectures
│   │   ├── train.py                   # Training pipeline
│   │   ├── evaluate.py                # CNN evaluation
│   │   ├── forward_propagation.py     # Forward pass implementations
│   │   ├── backward_propagation.py    # Backward pass implementations
│   │   ├── gradient_checker.py        # Gradient verification
│   │   ├── visualization.py           # Feature maps & Grad-CAM
│   │   ├── batch_inference.py         # Performance analysis
│   │   ├── utils.py                   # Utilities
│   │   ├── bonus_cnn.ipynb            # Bonus visualizations
│   │   └── cnn_eval.ipynb             # Evaluation notebook
│   │
│   ├── rnn/                           # RNN/LSTM implementations
│   │   ├── layers.py                  # RNN/LSTM cell definitions
│   │   ├── models.py                  # Captioning model
│   │   ├── train.py                   # Training
│   │   ├── evaluate.py                # Evaluation
│   │   ├── preprocessing.py           # Text preprocessing
│   │   └── rnn_lstm_scratch.ipynb     # RNN analysis
│   │
│   ├── pipeline/                      # End-to-end pipeline
│   │   ├── captioning_keras.py        # Keras-based captioning
│   │   ├── captioning_scratch.py      # From-scratch captioning
│   │   └── greedy_decode.py           # Decoding strategies
│   │
│   ├── evaluation/                    # Metrics & analysis
│   │   ├── metrics.py                 # BLEU-4, METEOR
│   │   └── qualitative.py             # Qualitative analysis
│   │
│   └── utils/                         # Utilities
│       └── weight_loader.py           # Model weight loading
│
├── results/                           # Output & results
│   ├── plots/                         # Generated visualizations
│   └── tables/                        # Results tables (JSON)
│
├── doc/                               # Documentation
└── experiments/                       # Experiment notebooks
```

---

## Pembagian Tugas Kelompok

### **NIM: 13523001**
**Nama**: Wardatul Khoiroh 
**Fokus**: CNN Implementation & Analysis

**Tanggung Jawab**:
- Membangun image loader dan data pipeline untuk Intel Image Dataset
- Mengimplementasikan komponen CNN from scratch: Conv2D, ReLU, MaxPooling, Flatten (NumPy-only)
- Melatih berbagai variasi CNN di Keras untuk menghasilkan reference weights
- Melakukan perbandingan performa antara shared weights (Conv2D) vs non-shared (LocallyConnected2D)
- Menghitung Macro F1 score dan analisis jumlah parameter
- Mengerjakan laporan bagian **Pembahasan (Discussion)** hasil analisis CNN

---

### **NIM: 13523018**
**Nama**: Raka Daffa Iftikhar
**Fokus**: RNN/LSTM Implementation & Captioning Models

**Tanggung Jawab**:
- Mengimplementasikan SimpleRNNCell dan LSTMCell dari scratch dengan semua gates (input, forget, cell, output)
- Membuat Embedding layer untuk token-to-vector conversion
- Melakukan preprocessing teks pada Flickr8k captions (tokenization, vocabulary building)
- Mengekstraksi fitur gambar menggunakan pretrained model (VGG16/InceptionV3) yang dibekukan
- Melatih minimal 12 variasi model decoder LSTM di Keras dengan berbagai hyperparameter
- Menyimpan weights untuk integrasi ke sistem from-scratch
- Mengerjakan laporan bagian **Hasil Pengujian (Results)** untuk RNN/LSTM
---

### **NIM: 13523053**
**Nama**: Sakti Bimasena  
**Fokus**: Integration, Evaluation & Repository Management

**Tanggung Jawab**:
- Menyatukan semua komponen (CNN + RNN) menjadi pipeline image captioning end-to-end
- Mengimplementasikan greedy decoding dan beam search strategies
- Menjalankan evaluasi pada semua model variasi menggunakan BLEU-4 dan METEOR metrics
- Melakukan analisis kualitatif: visualisasi predictions, failure cases analysis
- Mengelola struktur GitHub repository, version control, dan documentation
- Menyusun grafik training, hasil evaluasi, dan comparison charts
- Memastikan requirements.txt, environment.yml, dan seluruh laporan dokumentasi lengkap
- Mengerjakan laporan bagian **Kesimpulan dan Rekomendasi (Conclusion)**
---