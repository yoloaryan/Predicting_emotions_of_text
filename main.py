import os

# Limit TensorFlow memory and thread overhead for low-memory cloud hosts (e.g., Render free tier)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from contextlib import asynccontextmanager
from pathlib import Path
import re
import pickle

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH_H5 = BASE_DIR / "Artifacts" / "emotion_bigru_model.h5"
MODEL_PATH_KERAS = BASE_DIR / "Artifacts" / "emotion_bigru_model.keras"
MODEL_PATH = MODEL_PATH_H5 if MODEL_PATH_H5.exists() else MODEL_PATH_KERAS
TOKENIZER_PATH = BASE_DIR / "Artifacts" / "emotion_tokenizer.pkl"
STATIC_DIR = BASE_DIR / "static"

# ============================================================
# EMOTION LABELS
# IMPORTANT:
# These MUST match the label encoding used during training.
# ============================================================

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# ============================================================
# GLOBAL VARIABLES
# ============================================================

model = None
tokenizer = None
model_load_error = None

# ============================================================
# MODEL & TOKENIZER LOADER
# ============================================================


from tensorflow.keras.layers import InputLayer


class FixedInputLayer(InputLayer):

    def __init__(self, *args, **kwargs):
        batch_shape = kwargs.pop("batch_shape", None)
        kwargs.pop("optional", None)
        if batch_shape is not None and "input_shape" not in kwargs:
            kwargs["input_shape"] = batch_shape[1:]
        super().__init__(*args, **kwargs)


def load_model_and_tokenizer():
    global model
    global tokenizer
    global model_load_error

    if model is not None and tokenizer is not None:
        return True

    try:
        if model is None:
            print(f"Loading emotion model from {MODEL_PATH}...")
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file does not exist at {MODEL_PATH}")

            try:
                from tensorflow.keras.models import load_model as tf_load_model
                model = tf_load_model(MODEL_PATH, compile=False, custom_objects={"InputLayer": FixedInputLayer})
                print("Model loaded successfully using tensorflow.keras.")
            except Exception as e1:
                print(f"tf.keras load failed ({e1}), trying standalone keras...")
                import keras
                model = keras.models.load_model(MODEL_PATH, compile=False, custom_objects={"InputLayer": FixedInputLayer})
                print("Model loaded successfully using standalone keras.")

        if tokenizer is None:
            print(f"Loading tokenizer from {TOKENIZER_PATH}...")
            if not TOKENIZER_PATH.exists():
                raise FileNotFoundError(f"Tokenizer file does not exist at {TOKENIZER_PATH}")

            with open(TOKENIZER_PATH, "rb") as file:
                tokenizer = pickle.load(file)
            print("Tokenizer loaded successfully.")

        model_load_error = None
        print("Emotion AI is ready.")
        return True

    except Exception as e:
        import traceback
        err_msg = f"{e}\n{traceback.format_exc()}"
        print(f"ERROR while loading model/tokenizer: {err_msg}")
        model_load_error = str(e)
        return False


# ============================================================
# TEXT PREPROCESSING
# ============================================================


def clean_text(text: str) -> str:
    """
    Clean input text before sending it to the tokenizer.

    Keep preprocessing simple so that inference does not
    accidentally change the meaning of the input.
    """

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# LOAD MODEL + TOKENIZER LIFESPAN
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_and_tokenizer()
    yield
    print("Shutting down Emotion AI...")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Emotion AI API",
    description="Emotion Detection API using a Bidirectional GRU model",
    version="1.0.0",
    lifespan=lifespan)

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# REQUEST SCHEMA
# ============================================================


class PredictionRequest(BaseModel):
    text: str = Field(...,
                      min_length=1,
                      max_length=5000,
                      description="Text whose emotion needs to be predicted")


# ============================================================
# RESPONSE SCHEMA
# ============================================================


class PredictionResponse(BaseModel):
    text: str
    prediction_emotion: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    error: str | None = None


# ============================================================
# ROOT ENDPOINT
# ============================================================


@app.get("/")
async def home():

    index_file = STATIC_DIR / "index.html"

    if index_file.exists():
        return FileResponse(index_file)

    return {"message": "Emotion AI API is running", "docs": "/docs"}


# ============================================================
# HEALTH CHECK
# ============================================================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if model is None or tokenizer is None:
        load_model_and_tokenizer()

    return HealthResponse(status="Server is running",
                          model_loaded=model is not None
                          and tokenizer is not None,
                          error=model_load_error)



# ============================================================
# PREDICTION ENDPOINT
# ============================================================


@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(request: PredictionRequest):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if model is None or tokenizer is None:
        load_model_and_tokenizer()

    if model is None:
        raise HTTPException(status_code=503,
                            detail="Emotion model is not loaded.")

    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer is not loaded.")

    # --------------------------------------------------------
    # Get input text
    # --------------------------------------------------------

    original_text = request.text.strip()

    if not original_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    cleaned_text = clean_text(original_text)

    # --------------------------------------------------------
    # Convert text to sequence
    #
    # SAME tokenizer used during training
    # --------------------------------------------------------

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # --------------------------------------------------------
    # Pad sequence
    #
    # TRAINING USED:
    # maxlen = 50
    # padding = post
    # truncating = post
    # --------------------------------------------------------

    padded_sequence = pad_sequences(sequence,
                                    maxlen=50,
                                    padding="post",
                                    truncating="post")

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    prediction = model.predict(padded_sequence, verbose=0)

    # --------------------------------------------------------
    # Get probabilities
    # --------------------------------------------------------

    probabilities = prediction[0]

    # --------------------------------------------------------
    # Find highest probability class
    # --------------------------------------------------------

    predicted_index = int(np.argmax(probabilities))

    # --------------------------------------------------------
    # Convert class index to emotion
    #
    # 0 -> sadness
    # 1 -> joy
    # 2 -> love
    # 3 -> anger
    # 4 -> fear
    # 5 -> surprise
    # --------------------------------------------------------

    prediction_emotion = EMOTION_LABELS[predicted_index]

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = float(probabilities[predicted_index])

    # --------------------------------------------------------
    # All emotion probabilities
    # --------------------------------------------------------

    all_probabilities = {
        EMOTION_LABELS[i]: float(probabilities[i])
        for i in range(len(EMOTION_LABELS))
    }

    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return PredictionResponse(text=original_text,
                              prediction_emotion=prediction_emotion,
                              confidence=confidence,
                              all_probabilities=all_probabilities)


# ============================================================
# STATIC FRONTEND
# ============================================================

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

