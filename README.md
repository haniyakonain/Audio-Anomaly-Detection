🎧 FanSound AI: Machine Sound Anomaly Detection

FanSound AI is a web application that listens to machine audio and flags whether it sounds Normal or Abnormal. It combines a pretrained audio embedding model (YAMNet) with a custom-trained recurrent classifier (GRU) to catch early signs of mechanical trouble from sound alone, no extra sensors required.

Disclaimer: Predictions come from a machine learning model trained on a limited dataset and may not be 100% accurate. Use this as an early-warning aid alongside proper human inspection, not a replacement for it.

⚖️ Key Technologies and Libraries

🔧 Backend
- Flask: Python web framework serving the app and the `/predict` API.
- TensorFlow / Keras: Runtime for the trained anomaly-detection model.
- TensorFlow Hub (YAMNet): Pretrained audio embedding network; converts raw audio into a sequence of numeric fingerprints.
- librosa & soundfile: Audio loading, resampling, and decoding (wav/mp3/flac/ogg).
- scikit-learn & imbalanced-learn (SMOTE): Used during training to handle class imbalance between normal and abnormal samples.
- matplotlib: Generates the training/evaluation charts shown on the Charts page.
- gunicorn: Production WSGI server.

🖼 Frontend
- Server-rendered Jinja2 templates (no separate JS framework).
- Bootstrap 5 + a custom pastel design system (`static/css/style.css`).
- Vanilla JavaScript for tabs, drag-and-drop upload, and sample-library filtering (`static/js/app.js`).

🤖 Machine Learning
- YAMNet embeddings: Each audio clip is converted into a `(T, 1024)` sequence of embeddings.
- Bidirectional GRU classifier: Learns temporal patterns across the embedding sequence to output an Abnormal probability.
- Threshold tuning: Decision threshold picked during training and stored alongside the model (`model/best_threshold.txt`).

🌟 Features

🎙️ Detection
- Upload audio: Drag and drop a `.wav`, `.mp3`, `.flac`, or `.ogg` clip and get an instant Normal / Abnormal call.
- Sample audio library: Browse, preview, and predict on real bundled test recordings, no upload needed, filterable by machine ID and label.
- Result view: Probability gauge, decision threshold, and file name for every prediction.

📊 Charts page
Plain-language, data-grounded explanations of model performance, accuracy/loss curves, ROC curve with AUC, and a confusion matrix, written so a non-technical reader can follow what each chart means and what it shows for this model.

🏰 Project Structure
```
CODE/                         # Flask application (deployment root)
    app.py                    # Routes, feature extraction, prediction logic
    requirements.txt
    Procfile
    .python-version
    model/
        yamnet_gru_smote_tuned.keras
        yamnet_gru_smote_best.keras
        best_threshold.txt
    templates/                 # Jinja2 templates (base, index, predict, result, charts)
    static/
        css/style.css
        js/app.js
        charts/                 # Accuracy / Loss / ROC / Confusion Matrix PNGs
        favicon.ico
    fan_split_dataset/test/     # Bundled sample audio, grouped by machine ID and label
render.yaml                    # Render Blueprint (root deploy config)
```

⚙️ Installation and Setup

🔧 Local setup
Clone the repository:
```
git clone https://github.com/haniyakonain/Audio-Anomaly-Detection.git
cd Audio-Anomaly-Detection/CODE
```

Create and activate a Python virtual environment (Python 3.11 required, `tensorflow-cpu==2.15.0` does not support 3.12+):
```
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install required dependencies:
```
pip install -r requirements.txt
```

Run the Flask dev server:
```
python app.py
```

🚀 Running in production
```
gunicorn app:app --bind 0.0.0.0:$PORT
```

Then open a browser to `http://localhost:5000` (dev) or your deployed URL.

☁️ Deployment
The repo includes a `render.yaml` Blueprint for one-click deployment on [Render](https://render.com): New → Blueprint → select this repo → Apply. It pins Python 3.11.11, installs `CODE/requirements.txt`, and starts the app with gunicorn.

📊 Example Output

Input: `id_04/abnormal/00000124.wav` (bundled sample)
```
Prediction: Abnormal
Abnormal Probability: 55.4%
Threshold Used: 0.45
```

Input: `id_04/normal/00000949.wav` (bundled sample)
```
Prediction: Normal
Abnormal Probability: 41.3%
Threshold Used: 0.45
```

Model evaluation (held-out test set, 837 clips):
- Overall accuracy: 71.8%
- ROC AUC: 0.769
- Catches real faults (recall): 63.4%
- Clears real normals (specificity): 74.9%

Made with ❤️ by Haniya Konain
