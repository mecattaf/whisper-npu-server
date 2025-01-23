import librosa
import openvino_genai
from flask import Flask, request, jsonify
import io
import time
from pathlib import Path
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        # Create user-specific whisper directory
        self.user_home = os.path.expanduser('~')
        self.whisper_dir = os.path.join(self.user_home, '.whisper')
        self.models_dir = os.path.join(self.whisper_dir, 'models')
        self.model_name = "whisper-medium.en"
        self.pipeline = None
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        
    def initialize(self):
        """Initialize the model and prepare environment."""
        if self.pipeline is None:
            model_path = os.path.join(self.models_dir, self.model_name)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model {self.model_name} not found in {self.models_dir}. "
                    "Please ensure the model is properly downloaded."
                )
            
            logger.info(f"Loading model: {self.model_name}")
            
            try:
                self.pipeline = openvino_genai.WhisperPipeline(
                    str(model_path), 
                    device="NPU"
                )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to initialize model: {str(e)}")
                raise
        
        return self.pipeline

@app.route("/transcribe", methods=['POST'])
def transcribe():
    """Transcribe audio data."""
    try:
        pipeline = model_manager.pipeline
        if pipeline is None:
            return jsonify({"error": "Model not initialized"}), 500
        
        # Process the audio
        audio_data = request.get_data()
        if not audio_data:
            return jsonify({"error": "No audio data received"}), 400
            
        try:
            en_raw_speech, samplerate = librosa.load(io.BytesIO(audio_data), sr=16000)
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            return jsonify({"error": "Invalid audio data"}), 400
            
        try:
            genai_result = pipeline.generate(en_raw_speech)
            return jsonify({"text": str(genai_result)})
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            return jsonify({"error": "Transcription failed"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

# Initialize the model manager and load model at startup
model_manager = ModelManager()
model_manager.initialize()

if __name__ == '__main__':
    app.run(host='0.0.0.0')

