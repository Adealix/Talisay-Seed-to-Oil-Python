"""
Flask API Server for Talisay Oil Yield Prediction
Provides REST endpoints for the mobile/web application
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import numpy as np

sys.path.append(str(Path(__file__).parent))

from config import API_CONFIG
from predict import TalisayPredictor

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React Native app

# Set max content length
app.config["MAX_CONTENT_LENGTH"] = API_CONFIG["max_content_length"]

# Initialize predictor
predictor = TalisayPredictor(use_simple_color=True)


@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Talisay Oil Yield Prediction API",
        "version": "1.0.0"
    })


@app.route("/api/predict/image", methods=["POST"])
def predict_from_image():
    """
    Predict oil yield from uploaded image.
    
    Request body (JSON):
        - image: Base64 encoded image string
        - dimensions (optional): Known fruit dimensions
            - length_cm: float
            - width_cm: float
            - kernel_mass_g: float
            - whole_fruit_weight_g: float
    
    Returns:
        JSON with prediction results
    """
    try:
        data = request.get_json()
        
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode base64 image
        try:
            from PIL import Image
            
            image_data = data["image"]
            # Remove data URL prefix if present
            if "," in image_data:
                image_data = image_data.split(",")[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
        except Exception as e:
            return jsonify({"error": f"Invalid image data: {str(e)}"}), 400
        
        # Get optional dimensions
        known_dimensions = data.get("dimensions")
        
        # Analyze image
        result = predictor.analyze_image(image, known_dimensions)
        
        if not result["analysis_complete"]:
            return jsonify({
                "error": result.get("error", "Analysis failed"),
                "partial_result": result
            }), 500
        
        return jsonify({
            "success": True,
            "result": result
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/measurements", methods=["POST"])
def predict_from_measurements():
    """
    Predict oil yield from manual measurements.
    
    Request body (JSON):
        - color: "green", "yellow", or "brown"
        - length_cm: float
        - width_cm: float
        - kernel_mass_g: float (optional)
        - whole_fruit_weight_g: float (optional)
    
    Returns:
        JSON with prediction results
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields
        required = ["color", "length_cm", "width_cm"]
        missing = [f for f in required if f not in data]
        
        if missing:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing)}"
            }), 400
        
        # Validate color
        color = data["color"].lower()
        if color not in ["green", "yellow", "brown"]:
            return jsonify({
                "error": "Invalid color. Must be 'green', 'yellow', or 'brown'"
            }), 400
        
        # Analyze
        result = predictor.analyze_measurements(
            color=color,
            length_cm=float(data["length_cm"]),
            width_cm=float(data["width_cm"]),
            kernel_mass_g=data.get("kernel_mass_g"),
            whole_fruit_weight_g=data.get("whole_fruit_weight_g")
        )
        
        return jsonify({
            "success": True,
            "result": result
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/research", methods=["GET"])
def get_research_data():
    """
    Get scientific research data used for predictions.
    
    Returns:
        JSON with research parameters and references
    """
    try:
        research = predictor.get_research_summary()
        return jsonify({
            "success": True,
            "data": research
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/color-guide", methods=["GET"])
def get_color_guide():
    """
    Get color classification guide for users.
    
    Returns:
        JSON with color descriptions and oil yield expectations
    """
    from config import OIL_YIELD_BY_COLOR
    
    guide = {
        "colors": [
            {
                "name": "green",
                "display_name": "Green (Immature)",
                "description": "The fruit is not yet ripe. The skin is predominantly green.",
                "oil_yield_range": f"{OIL_YIELD_BY_COLOR['green']['min']}-{OIL_YIELD_BY_COLOR['green']['max']}%",
                "recommendation": "Wait for the fruit to mature for higher oil yield.",
                "hex_color": "#4CAF50"
            },
            {
                "name": "yellow",
                "display_name": "Yellow (Mature)",
                "description": "The fruit is ripe and at optimal stage for oil extraction.",
                "oil_yield_range": f"{OIL_YIELD_BY_COLOR['yellow']['min']}-{OIL_YIELD_BY_COLOR['yellow']['max']}%",
                "recommendation": "Best time to harvest for maximum oil yield!",
                "hex_color": "#FFC107"
            },
            {
                "name": "brown",
                "display_name": "Brown (Fully Ripe)",
                "description": "The fruit is fully ripe or overripe with brownish skin.",
                "oil_yield_range": f"{OIL_YIELD_BY_COLOR['brown']['min']}-{OIL_YIELD_BY_COLOR['brown']['max']}%",
                "recommendation": "Still good for extraction, but yellow stage is optimal.",
                "hex_color": "#795548"
            }
        ],
        "summary": {
            "best_color": "yellow",
            "reason": "Yellow (mature) fruits have the highest oil content based on scientific research."
        }
    }
    
    return jsonify({
        "success": True,
        "data": guide
    })


@app.route("/api/dimensions-guide", methods=["GET"])
def get_dimensions_guide():
    """
    Get guide for measuring fruit dimensions.
    
    Returns:
        JSON with measurement instructions and typical ranges
    """
    from config import DIMENSION_RANGES
    
    guide = {
        "measurements": [
            {
                "name": "length_cm",
                "display_name": "Fruit Length",
                "unit": "centimeters (cm)",
                "typical_range": f"{DIMENSION_RANGES['length']['min']} - {DIMENSION_RANGES['length']['max']} cm",
                "how_to_measure": "Measure from the stem end to the tip of the fruit along the longest axis.",
                "importance": "Length correlates moderately with oil yield."
            },
            {
                "name": "width_cm",
                "display_name": "Fruit Width",
                "unit": "centimeters (cm)",
                "typical_range": f"{DIMENSION_RANGES['width']['min']} - {DIMENSION_RANGES['width']['max']} cm",
                "how_to_measure": "Measure the widest point of the fruit perpendicular to the length.",
                "importance": "Width helps estimate fruit volume and weight."
            },
            {
                "name": "kernel_mass_g",
                "display_name": "Kernel Mass",
                "unit": "grams (g)",
                "typical_range": f"{DIMENSION_RANGES['kernel_mass']['min']} - {DIMENSION_RANGES['kernel_mass']['max']} g",
                "how_to_measure": "Remove the outer flesh and weigh the inner kernel/seed.",
                "importance": "MOST IMPORTANT - kernel mass is the strongest predictor of oil yield!"
            },
            {
                "name": "whole_fruit_weight_g",
                "display_name": "Whole Fruit Weight",
                "unit": "grams (g)",
                "typical_range": f"{DIMENSION_RANGES['whole_fruit_weight']['min']} - {DIMENSION_RANGES['whole_fruit_weight']['max']} g",
                "how_to_measure": "Weigh the entire fruit including flesh and kernel.",
                "importance": "Helps calculate kernel-to-fruit ratio."
            }
        ],
        "tips": [
            "Use a digital caliper or ruler for length and width",
            "Use a kitchen scale for accurate weight measurements",
            "Kernel mass has the highest correlation with oil yield",
            "If you can't measure kernel mass, the app will estimate it"
        ]
    }
    
    return jsonify({
        "success": True,
        "data": guide
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({
        "error": "File too large. Maximum size is 16MB."
    }), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server error."""
    return jsonify({
        "error": "Internal server error"
    }), 500


if __name__ == "__main__":
    print("Starting Talisay Oil Yield Prediction API...")
    print(f"Server running on http://{API_CONFIG['host']}:{API_CONFIG['port']}")
    print("\nEndpoints:")
    print("  GET  /                     - Health check")
    print("  POST /api/predict/image    - Predict from image")
    print("  POST /api/predict/measurements - Predict from measurements")
    print("  GET  /api/research         - Get research data")
    print("  GET  /api/color-guide      - Get color classification guide")
    print("  GET  /api/dimensions-guide - Get measurement guide")
    
    app.run(
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        debug=API_CONFIG["debug"]
    )
