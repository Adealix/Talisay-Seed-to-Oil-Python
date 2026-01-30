# React Native + Python ML Integration Guide

## Overview
The React Native Expo app is now integrated with the Python ML backend for Talisay fruit analysis.

## Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    React Native Expo App                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ ScanScreen.js                                            │   │
│  │   - Captures/selects images                              │   │
│  │   - Displays ML results (dimensions, coin detection)    │   │
│  │   - Shows oil yield prediction                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ mlService.js                                             │   │
│  │   - Encodes image to base64                              │   │
│  │   - Calls Flask API endpoints                            │   │
│  │   - Transforms ML results for app                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP (JSON)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Python ML Backend                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ api.py (Flask)                                           │   │
│  │   - /api/predict/image - Analyze image                   │   │
│  │   - /api/predict/measurements - Manual input             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ predict.py                                               │   │
│  │   - Fruit validation (Talisay detection)                 │   │
│  │   - Coin detection (₱5 coin = 25mm reference)            │   │
│  │   - Color segmentation (green/yellow/brown)              │   │
│  │   - Dimension estimation                                 │   │
│  │   - Oil yield prediction (TensorFlow model)              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Files Modified/Created

### New Files
- `src/services/mlService.js` - ML backend API service

### Updated Files
- `src/screens/ScanScreen.js` - Added ML integration, results display
- `src/services/index.js` - Export mlService
- `.env` - Added EXPO_PUBLIC_ML_API_URL

## How to Test

### Step 1: Start the Python ML Backend
```bash
cd d:\talisay_ml
.\.venv\Scripts\Activate.ps1
python ml/api.py
```
The server will start at `http://localhost:5000`

### Step 2: Start the Expo App
In a new terminal:
```bash
cd d:\talisay_ml
npx expo start
```

### Step 3: Test the Integration
1. Open the app on your device/emulator
2. Go to "Scan Fruit" screen
3. You should see:
   - "ML Backend Connected" indicator (green) if Flask server is running
   - Photo tip about placing ₱5 coin on left, fruit on right
4. Pick an image from gallery or take a photo
5. The app will:
   - Send image to ML backend as base64
   - Display analysis results:
     - Predicted oil yield percentage
     - Detected dimensions (length, width, weight)
     - Coin detection status
     - Interpretation message

## Photo Guidelines for Best Results
1. Place a ₱5 silver coin on the LEFT side of the image
2. Place the Talisay fruit on the RIGHT side
3. Ensure good lighting (avoid shadows)
4. Keep camera perpendicular to the surface
5. Use a contrasting background (not green/yellow/brown)

## Configuration

### For Android Emulator
Update `.env`:
```
EXPO_PUBLIC_ML_API_URL=http://10.0.2.2:5000
```

### For Physical Device
Find your computer's IP address and update `.env`:
```
EXPO_PUBLIC_ML_API_URL=http://192.168.x.x:5000
```

Also start Flask with host binding:
```bash
python ml/api.py --host 0.0.0.0
```

## Fallback Behavior
If the ML backend is unavailable:
- The app automatically falls back to local color analysis
- Manual category selection is still available
- Oil yield is estimated using the local `predictRatio` function

## API Endpoints Used

### POST /api/predict/image
Request:
```json
{
  "image": "<base64 encoded image>",
  "dimensions": { // optional
    "length_cm": 5.0,
    "width_cm": 3.5
  }
}
```

Response:
```json
{
  "success": true,
  "result": {
    "analysis_complete": true,
    "detected_color": "yellow",
    "reference_detected": true,
    "dimensions": {
      "length_cm": 5.2,
      "width_cm": 3.8,
      "whole_fruit_weight_g": 45.0
    },
    "oil_yield_percent": 38.5,
    "interpretation": "Yellow Talisay with moderate oil content..."
  }
}
```
