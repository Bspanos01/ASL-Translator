# ASL Translator + FREE-WILi Display

Real-time American Sign Language letter recognition with AI-powered word autocomplete, text-to-speech, and hardware display output.

**Team:** Bill Spanos & Jayden — GrizzHacks 8
**Tracks:** AI/ML, Social Good

## Features

- Real-time hand tracking with MediaPipe (21 landmarks)
- ASL letter classification (A–Z) via RandomForest (~96% accuracy)
- Debounce system to prevent jitter — hold a sign briefly to confirm
- Word autocomplete with offline dictionary (500+ words) and optional Gemini API
- Text-to-speech — letters spoken as signed, full sentence spoken on word confirm
- Sentence building across confirmed words
- FREE-WILi hardware display output over USB (optional)
- Clean overlay UI with hand skeleton, confidence bars, and suggestion panel

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Configure Gemini API key for AI-powered autocomplete
cp .env.example .env
# Edit .env and add your Gemini API key
# The app works fully offline without it

# 3. Download dataset and train model
# Get the ASL Alphabet dataset from:
# https://www.kaggle.com/datasets/grassknoted/asl-alphabet
# Extract into data/, then run:
python3 data/preprocess_images.py
python3 model/train.py

# 4. Run the translator
python3 main.py
```

## Controls

| Key | Action |
|-----|--------|
| **SPACE** | Confirm current word + speak sentence |
| **D** | Delete last letter |
| **1 / 2 / 3** | Pick autocomplete suggestion |
| **S** | Speak entire sentence |
| **J** | Insert letter J |
| **Z** | Insert letter Z |
| **C** | Clear all (reset everything) |
| **X** | Remove last confirmed word |
| **ESC** | Quit |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Hand Tracking | MediaPipe Hands |
| Classification | scikit-learn RandomForest |
| Autocomplete | Offline dictionary + Gemini API (optional) |
| Text-to-Speech | macOS `say` command |
| Display | OpenCV |
| Hardware Output | FREE-WILi USB display |
| Language | Python 3.12 |

## File Structure

```
asl-translator/
├── main.py                 # Application entry point
├── requirements.txt
├── .env.example
├── .gitignore
├── core/
│   ├── hand_tracker.py     # MediaPipe hand landmark extraction
│   ├── classifier.py       # ASL letter prediction
│   ├── letter_buffer.py    # Debounce + word buffering
│   ├── autocomplete.py     # Offline + Gemini API word suggestions
│   ├── speaker.py          # Text-to-speech output
│   └── projector.py        # FREE-WILi hardware interface
├── model/
│   ├── train.py            # Model training script
│   └── evaluate.py         # Evaluation + confusion matrix
├── ui/
│   └── display.py          # OpenCV overlay rendering
└── data/
    ├── preprocess_images.py  # Convert Kaggle images to landmark CSVs
    └── collect.py            # Webcam data collection tool
```

## FREE-WILi Integration

The app automatically detects a connected FREE-WILi device over USB. If no device is found, the app runs normally with on-screen display only. When connected, recognized letters, confirmed words, and sentences are pushed to the FREE-WILi screen in real time, and LEDs flash on word confirmation.

## Dataset

Training data: [ASL Alphabet Dataset (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
