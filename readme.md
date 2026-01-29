# 🧠 Alzheimer MRI Detection

An end-to-end deep learning project for detecting Alzheimer’s disease stages from MRI scans using a Custom CNN, FastAPI, Streamlit, SQLite, pytest, and Docker.

## 🏗️ Architecture

| 📂 Data | ➡️ | 🔧 Preprocessing | ➡️ | 🧠 Model Training | ➡️ | 📊 Evaluation | ➡️ | ⚡ Serving | ➡️ | 🐳 Deployment |
|---------|------|------------------|------|-------------------|------|---------------|------|-----------|------|---------------|
| MRI images | ➡️| Resizing, normalization, augmentation |➡️ | Custom CNN, EfficientNetB0, DenseNet121 | ➡️| Accuracy, AUC, loss, F1-score, MCC |➡️ | FastAPI (predict, save, history), Streamlit UI |➡️ | Docker (API + Streamlit), CI with GitHub Actions |


## 🎯 Overview

- **Task**: Classify MRI scans into four Alzheimer’s stages: Non Demented, Very Mild Demented, Mild Demented, Moderate Demented.
- **Data**: MRI images in `train/` and `test/` folders.
- **Goal**: Predict stages, save records, and provide a user-friendly interface for doctors.

## 🚀 Features

- **Model**: Custom CNN, compared with EfficientNetB0 and DenseNet121.
- **FastAPI Backend**:
  - `POST /predict`: Predict stage from MRI.
  - `POST /save_record`: Save results to SQLite.
  - `GET /history/{patient_id}`: Retrieve patient history.
- **Streamlit Frontend**: Upload images, view predictions, and export PDF reports to the `output/` directory.

  ![Streamlit Interface with Prediction Results](img/streamlit_screenshot.png)
- **Database**: SQLite for patient records.
- **Testing**: Pytest for API and preprocessing.
- **Containerization**: Docker and Docker Compose.

## 📊 Model Performance

| Model          | Accuracy | Loss | AUC  | F1-score | MCC  |
|----------------|----------|------|------|----------|------|
| Custom CNN     | **0.97**     | **0.07** | **0.99** | **0.98**     | **0.96** |
| EfficientNetB0 | 0.95     | 0.12 | 0.99 | 0.95     | 0.92 |
| DenseNet121    | 0.76     | 0.52 | 0.94 | 0.76     | 0.61 |

> **Note**: ⭐ Custom CNN selected for deployment.

## 🛠️ Technologies

- **Deep Learning**: TensorFlow, Keras
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Database**: SQLite
- **Testing**: Pytest
- **Containerization**: Docker, Docker Compose

## 📦 Installation

1. **Clone Repository**:
   ```bash
   git clone https://github.com/yourusername/alzheimer_mri_detection.git
   cd alzheimer_mri_detection
   ```

2. **Download Dataset**:
   - Get from: [Best Alzheimer MRI Dataset](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy)
   - After downloading and extracting the archive (which may be named `archive.zip`), locate the folder named `combined data`. Inside it, you will find the `train/` and `test/` directories.

    - Copy or move these `train/` and `test/` folders directly into the `data/` directory of this repository.

    Your project directory structure should be like:

    ```
    data/
    ├── train/
    │   ├── (category folders with training images)
    ├── test/
    │   ├── (category folders with testing images)
    ```
3. **Run Options**:

   **Option A: Local**
   - Create and activate a virtual environment:
     ```bash
     # On macOS/Linux
     python3 -m venv venv
     source venv/bin/activate

     # On Windows
     python -m venv venv
     venv\Scripts\activate
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Run FastAPI:
     ```bash
     uvicorn api:app --host 0.0.0.0 --port 8001
     ```
     API at [http://localhost:8001/docs](http://localhost:8001/docs).
   - Run Streamlit:
     ```bash
     streamlit run app.py
     ```
     UI at [http://localhost:8501](http://localhost:8501).

   **Option B: Docker**
   - Build and run:
     ```bash
     docker compose up --build
     ```
     - API: [http://localhost:8001](http://localhost:8001)
     - Streamlit: [http://localhost:8501](http://localhost:8501)
   - Detached mode:
     ```bash
     docker compose up -d
     ```
   - Stop:
     ```bash
     docker compose down
     ```

## 🧪 Testing

Run unit tests:
```bash
pytest
```
## 📂 Data Attribution

This project uses the [Best Alzheimer MRI Dataset (99% Accuracy)](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy) from Kaggle, licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Attribution: Dataset by Luke Chugh (© 2023). See the [LICENSE](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy/license) for full terms.


## 📚 References

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [TensorFlow Docs](https://www.tensorflow.org/)
- [Docker Python Guide](https://docs.docker.com/language/python/)

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.