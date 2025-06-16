# 🐾 Dog & Cat Detector 🐾

Welcome to the **Dog & Cat Detector** — a web application built with **Django** and powered by a **Convolutional Neural Network (CNN)** to classify images as either a **Dog** or a **Cat**.

---

## 📌 What this project does

This web app allows users to upload an image of a cat or dog, and it predicts whether the image contains a **cat** or **dog** using a deep learning model built with **TensorFlow**.

---

## 🚀 How to run the Django app

1️⃣ **Clone this repository**

```bash
git clone https://github.com/YourUsername/CatDetectDog.git
cd CatDetectDog
```

2️⃣ **Set up a virtual environment (Recommended)**

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

*(If `requirements.txt` isn’t available yet, you can generate it by running: `pip freeze > requirements.txt` after setting up your environment)*

4️⃣ **Run the Django server**

```bash
python manage.py runserver
```

5️⃣ Open your browser and go to:

```
http://127.0.0.1:8000/
```

---

## 🧠 How to train the model

1️⃣ **Prepare the dataset folder:**

```
dataset/
🔗 train/
🔗 cats/
🔗 dogs/
🔗 validation/
🔗 cats/
🔗 dogs/
```

️⚠️ *Dataset is not included in this repository. Prepare your own dataset of images.*

2️⃣ **Run the training script**

```bash
python train_model.py
```

3️⃣ **After training finishes, your trained model will be saved as:**

```
dog_cat_model.keras
```

👉 Place this model in the Django project directory or adjust your code accordingly.

---

## ✨ Features

* Simple & beautiful upload interface
* Real-time image predictions
* Confusion Matrix & Accuracy graphs during training
* Customizable — train with your own dataset!

---

## ⚠️ Notes

* The dataset and pre-trained model are **NOT included** in this repository.
* Refer to `NOTES.txt` for more instructions on dataset preparation.

---

## 📧 Contact

Created by /Andrew Hama Mutamiri • Open for contributions and improvements!
