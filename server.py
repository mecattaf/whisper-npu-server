import librosa
import openvino_genai
from flask import Flask, request, jsonify
import io
import os
from pathlib import Path
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.user_home = os.path.expanduser('~')
        self.whisper_dir = os.path.join(self.user_home, '.whisper')
        self.models_dir = os.path.join(self.whisper_dir, 'models')
        self.pipelines = {}
        self.default_model = "whisper-small"
        
        os.makedirs(self.models_dir, exist_ok=True)
        
    def load_model(self, model_name):
        if model_name not in self.pipelines:
            model_path = os.path.join(self.models_dir, model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model {model_name} not found")
            
            logger.info(f"Loading model: {model_name}")
            self.pipelines[model_name] = openvino_genai.WhisperPipeline(str(model_path), device="NPU")
        
        return self.pipelines[model_name]

    def list_models(self):
        return [d for d in os.listdir(self.models_dir) 
                if os.path.isdir(os.path.join(self.models_dir, d))]

@app.route("/models", methods=['GET'])
def list_models():
    return jsonify({"models": model_manager.list_models()})

@app.route("/transcribe/<model_name>", methods=['POST'])
def transcribe_with_model(model_name):
    try:
        pipeline = model_manager.load_model(model_name)
        audio_data = request.get_data()
        if not audio_data:
            return jsonify({"error": "No audio data"}), 400
            
        en_raw_speech, _ = librosa.load(io.BytesIO(audio_data), sr=16000)
        genai_result = pipeline.generate(en_raw_speech)
        return jsonify({"text": str(genai_result)})
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/transcribe", methods=['POST'])
def transcribe():
    return transcribe_with_model(model_manager.default_model)

model_manager = ModelManager()
model_manager.load_model(model_manager.default_model)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
