# 🌿 Plant Defect Identification

This project uses a Convolutional Neural Network (CNN) to detect and classify plant leaf defects. Early identification of plant diseases is crucial in agriculture, and this tool helps automate that process using image classification.

---

## 🚀 Features

- Image classification
- Multiple plant disease classes supported
- Easily extendable and ready for deployment
- Clean and simple structure

---

## 📁 Project Structure

```

Plant-defect-identification/
├── data/                 # Sample data or images
├── main.py              # Script to run inference on plant images
├── class\_indices.json   # Class label mapping
├── requirements.txt     # Dependencies
├── .gitignore
└── model.h5             # (Not included here - see below)

````

---

## 🧠 Trained Model File

The trained model file `model.h5` is not included in this GitHub repo due to size limitations.

🔗 **[Download the model here](https://drive.google.com/your-link-here)**  
*(Replace this link with your actual Google Drive or Dropbox link)*

Once downloaded, place `model.h5` in the root folder of the project.

---

## 🛠 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/saniya-154/Plant-defect-identification.git
cd Plant-defect-identification
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the model

```bash
python main.py
```

The script will load the image, process it, and print the predicted plant disease class.

---

## 🧾 Classes Detected

The model can classify the following (example classes):

* Healthy
* Leaf Spot
* Leaf Rust
* Powdery Mildew

*(Edit this list based on your actual dataset)*

---

## 💡 Future Scope

* Real-time prediction from webcam
* Streamlit web app interface
* Extend dataset with more plant species

---

## 👩‍💻 Author

Made by [Saniya Sayyed](https://github.com/saniya-154)

---

## 📄 License

This project is licensed under the MIT License.

```

---

Once you upload your `.h5` model to Google Drive or Dropbox:
- Replace the placeholder link in the **“Download the model here”** section.

Let me know if you want help writing the `main.py` usage section in more detail or turning this into a Streamlit app!
```
