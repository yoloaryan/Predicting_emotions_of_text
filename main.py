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
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Artifacts" / "emotion_bigru_model.keras"
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
# LOAD MODEL + TOKENIZER
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global tokenizer

    try:
        print("Loading emotion model...")

        model = load_model(MODEL_PATH)

        print("Model loaded successfully.")

        print("Loading tokenizer...")

        with open(TOKENIZER_PATH, "rb") as file:
            tokenizer = pickle.load(file)

        print("Tokenizer loaded successfully.")

        print("Emotion AI is ready.")

    except Exception as e:
        print(f"ERROR while loading model/tokenizer: {e}")

        model = None
        tokenizer = None

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

    return HealthResponse(status="Server is running",
                          model_loaded=model is not None
                          and tokenizer is not None)


# ============================================================
# PREDICTION ENDPOINT
# ============================================================


@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(request: PredictionRequest):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

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
