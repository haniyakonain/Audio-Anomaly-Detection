# app.py
from pathlib import Path
from flask import (
    Flask, render_template, request, redirect, url_for,
    send_from_directory, flash, abort
)
from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa

# -----------------------------
# Configuration
# -----------------------------
APP_ROOT = Path(__file__).parent.resolve()
MODEL_DIR = APP_ROOT / "model"
UPLOAD_DIR = APP_ROOT / "uploads"
STATIC_CHARTS = APP_ROOT / "static" / "charts"
SAMPLE_DIR = APP_ROOT / "fan_split_dataset" / "test"

MODEL_FILE = MODEL_DIR / "yamnet_gru_smote_tuned.keras"
THRESH_FILE = MODEL_DIR / "best_threshold.txt"

ALLOWED_EXT = {".wav", ".flac", ".ogg", ".mp3"}
MAX_SEQ = 100   # must match training
TARGET_SR = 16000
SAMPLES_PER_GROUP = 8  # cap how many sample files we surface per machine/class

# create directories if missing
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STATIC_CHARTS.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Flask app init
# -----------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "replace_this_with_a_random_secret"  # only needed for flash()

# -----------------------------
# Load ML assets at startup
# -----------------------------
# Load Keras model
from keras.saving import load_model  # important fix for Keras 3

try:
    model = load_model(
        str(MODEL_FILE),
        compile=False,
        safe_mode=False
    )
    print("[INFO] Model loaded successfully with custom loader.")
except Exception as e:
    print("[ERROR] Failed to load model:", e)
    model = None


# Load threshold
if not THRESH_FILE.exists():
    print(f"[WARN] Threshold file not found at {THRESH_FILE}. Using default 0.5")
    best_threshold = 0.5
else:
    try:
        best_threshold = float(THRESH_FILE.read_text().strip())
    except Exception as e:
        print("[WARN] Failed to read threshold file:", e)
        best_threshold = 0.5
print(f"[INFO] Using threshold = {best_threshold}")

# Load yamnet from TF Hub once
print("[INFO] Loading YAMNet from TF Hub (this may take a few seconds)...")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
print("[INFO] YAMNet loaded.")

# -----------------------------
# Utility functions
# -----------------------------
def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT

def load_audio_file(path, target_sr=TARGET_SR):
    """Load audio and return 1D float32 waveform at target_sr."""
    audio, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)

def yamnet_sequence_embeddings(waveform):
    """Return numpy array of YAMNet embeddings (T, 1024) for a 1D waveform."""
    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet(waveform_tf)
    return embeddings.numpy()

def pad_sequence(seq, max_len=MAX_SEQ):
    """Pad or trim a (T, D) sequence to (max_len, D)."""
    T, D = seq.shape
    if T >= max_len:
        return seq[:max_len]
    else:
        pad_len = max_len - T
        pad = np.zeros((pad_len, D), dtype=np.float32)
        return np.vstack([seq, pad])

def extract_features_from_path(filepath):
    """Load audio, extract yamnet embeddings, pad to MAX_SEQ and return shape (1, MAX_SEQ, 1024)."""
    waveform = load_audio_file(str(filepath))
    emb = yamnet_sequence_embeddings(waveform)   # (T, 1024)
    emb_pad = pad_sequence(emb, max_len=MAX_SEQ) # (MAX_SEQ, 1024)
    return np.expand_dims(emb_pad, axis=0)       # (1, MAX_SEQ, 1024)

def list_sample_audio():
    """Group demo audio files bundled in fan_split_dataset/test by machine id + class.

    Returns a list of {id, label, files: [{name, relpath}], total} dicts,
    capped to SAMPLES_PER_GROUP files per group so the library stays browsable.
    """
    groups = []
    if not SAMPLE_DIR.exists():
        return groups

    for machine_dir in sorted(SAMPLE_DIR.iterdir()):
        if not machine_dir.is_dir():
            continue
        for label_dir in sorted(machine_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            wavs = sorted(
                f for f in label_dir.iterdir()
                if f.suffix.lower() in ALLOWED_EXT
            )
            if not wavs:
                continue
            shown = wavs[:SAMPLES_PER_GROUP]
            groups.append({
                "machine_id": machine_dir.name,
                "label": label_dir.name,  # "normal" / "abnormal"
                "total": len(wavs),
                "files": [
                    {
                        "name": f.name,
                        "relpath": f.relative_to(SAMPLE_DIR).as_posix(),
                    }
                    for f in shown
                ],
            })
    return groups

def resolve_sample_path(relpath):
    """Safely resolve a relpath (as handed back to the client) inside SAMPLE_DIR."""
    candidate = (SAMPLE_DIR / relpath).resolve()
    sample_root = SAMPLE_DIR.resolve()
    if sample_root not in candidate.parents and candidate != sample_root:
        return None
    if not candidate.is_file():
        return None
    return candidate

def run_prediction(filepath, display_name):
    """Shared prediction path for both uploads and sample-library picks."""
    if model is None:
        return None, "Model is not loaded. Please place the model in model/ folder."
    try:
        X = extract_features_from_path(filepath)
        prob = float(model.predict(X, verbose=0).ravel()[0])
        label = "Abnormal" if prob >= best_threshold else "Normal"
        return {"filename": display_name, "prob": prob, "label": label,
                "threshold": best_threshold}, None
    except Exception as e:
        app.logger.exception("Prediction error")
        return None, f"Error during prediction: {e}"

# -----------------------------
# Routes: Home / Pages
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------
# Prediction page
# -----------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        sample_relpath = request.form.get("sample_path", "").strip()

        # Option 1: user picked a file from the bundled sample library
        if sample_relpath:
            filepath = resolve_sample_path(sample_relpath)
            if filepath is None:
                flash("Invalid sample selection.", "danger")
                return redirect(url_for("predict"))
            result, error = run_prediction(filepath, filepath.name)
            if error:
                flash(error, "danger")
                return redirect(url_for("predict"))
            return render_template("result.html", **result)

        # Option 2: user uploaded their own file
        if "audio" not in request.files:
            flash("No file part in the request.", "danger")
            return redirect(request.url)

        file = request.files["audio"]
        if file.filename == "":
            flash("No selected file.", "warning")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Unsupported file type. Allowed: wav, ogg, flac, mp3", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        save_path = UPLOAD_DIR / filename
        file.save(save_path)

        result, error = run_prediction(save_path, filename)

        try:
            save_path.unlink(missing_ok=True)
        except Exception:
            pass

        if error:
            flash(error, "danger")
            return redirect(request.url)
        return render_template("result.html", **result)

    return render_template("predict.html", sample_groups=list_sample_audio())


# -----------------------------
# Serve bundled sample audio for in-browser preview
# -----------------------------
@app.route("/sample-audio/<path:relpath>")
def sample_audio(relpath):
    filepath = resolve_sample_path(relpath)
    if filepath is None:
        abort(404)
    return send_from_directory(str(SAMPLE_DIR), relpath)


# -----------------------------
# Charts page (serves static images if present)
# -----------------------------
@app.route("/charts")
def charts():
    # list chart image files in static/charts
    chart_files = []
    if STATIC_CHARTS.exists():
        for f in STATIC_CHARTS.iterdir():
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
                chart_files.append(f.name)
    return render_template("charts.html", charts=chart_files)


@app.route("/static/charts/<path:filename>")
def send_chart(filename):
    return send_from_directory(str(STATIC_CHARTS), filename)


# -----------------------------
# Simple health check
# -----------------------------
@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "threshold": best_threshold}, 200


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(host="0.0.0.0", port=5000, debug=True)
