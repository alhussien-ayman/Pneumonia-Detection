# ClearLung — Pneumonia Detection from Chest X-Rays

Classical ML pipeline for binary pneumonia classification.  

## Step-by-Step Setup & Run Guide

### Step 1 — Clone the Repository
```bash
git clone https://github.com/alhussien-ayman/Pneumonia-Detection
cd Pneumonia-Detection 
```

---

### Step 2 — Create Virtual Environment
```bash
python -m venv venv
```

---

### Step 3 — Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\Activate
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal line.

---

### Step 4 — Install Dependencies
```bash
pip install -r requirements.txt
```

---

### Step 5 — Get Kaggle API Key

1. Go to https://www.kaggle.com
2. Click your profile picture → Settings
3. Scroll to API section → Click Create New Token
4. A file called kaggle.json will download
5. Place it here:

**Windows:**
```
C:\Users\YourName\.kaggle\kaggle.json
```

**Mac / Linux:**
```
~/.kaggle/kaggle.json
```

---

### Step 6 — Download Dataset
```bash
python src/download_data.py
```

This will automatically download and extract the dataset into data/:
```
data/
  chest_xray/
    train/
      NORMAL/        (1,341 images)
      PNEUMONIA/     (3,875 images)
    val/
      NORMAL/        (8 images)
      PNEUMONIA/     (8 images)
    test/
      NORMAL/        (234 images)
      PNEUMONIA/     (390 images)
```

