# 🧴 Skin Disease Classifier using CNN

## 🧠 Project Overview
This project is a **Skin Disease Classifier** that identifies skin conditions from dermoscopic images.  
It classifies images into categories such as **Eczema**, **Melanoma**, **Acne**, **Psoriasis**, **Rosacea**, and **Healthy Skin** using a **Convolutional Neural Network (CNN)**.

---

## 🚀 Features
- Predicts **skin disease** from uploaded images  
- Accepts **JPG, JPEG, PNG** image formats  
- Displays **top 3 predictions** with confidence percentages  
- Simple **Streamlit interface** for easy interaction  
- Stores **trained model** in `.h5` format  
- Supports **model retraining** on new datasets  

---

## 🧩 Libraries Used
- **tensorflow** – For building and training the CNN model  
- **keras** – For defining model layers and training utilities  
- **numpy** – For numerical computations  
- **Pillow (PIL)** – For image loading and preprocessing  
- **streamlit** – For creating the interactive web app  

---

## ⚙️ How It Works
1. **Model Creation:**  
   - Defines a **CNN architecture** with Conv2D, MaxPooling2D, Flatten, and Dense layers.  
2. **Training:**  
   - Uses a labeled dataset of dermoscopic images to train the CNN.  
   - Saves the trained model as `best_model.h5`.  
3. **Image Upload:**  
   - Users upload a skin image via the Streamlit interface.  
4. **Preprocessing:**  
   - Resizes images to 224×224 and normalizes pixel values.  
5. **Prediction:**  
   - Model predicts probabilities for each skin condition.  
   - Displays the **top 3 predictions** with confidence scores.  
6. **Class Mapping:**  
   - Labels are loaded from `labels.txt` if not embedded in the model.  

---

## 📊 Sample Output
**Uploaded Image:** ![example](example.png)  

**Predictions (Top 3):**  
- Melanoma: 78.45%  
- Eczema: 12.34%  
- Acne: 5.67%  

---

## 💻 Technologies Used
| Category | Technology |
|-----------|-------------|
| Programming Language | Python |
| Framework | Streamlit |
| Deep Learning | CNN (Convolutional Neural Network) |
| Image Processing | Pillow, NumPy |
| Model Storage | TensorFlow / Keras `.h5` |

---

## 🧰 Installation & Setup
1. **Create Virtual Environment**  
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# or
source .venv/bin/activate  # macOS/Linux
