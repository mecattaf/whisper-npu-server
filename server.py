
import librosa
import openvino_genai

from flask import Flask
from flask import request

import io
import time

ov_pipe = openvino_genai.WhisperPipeline("/models", device="NPU")

app = Flask(__name__)

@app.route("/transcribe", methods=['POST'])
def hello():
    audio_data = request.get_data()
    app.logger.info("Received audio data %s bytes", len(audio_data))
    en_raw_speech, samplerate = librosa.load(io.BytesIO(audio_data), sr=16000)
    audio_duration = len(en_raw_speech) / samplerate

    start = time.time_ns() / 1_000_000_000

    app.logger.info("Starting inference")

    genai_result = ov_pipe.generate(en_raw_speech)

    end = time.time_ns() / 1_000_000_000

    app.logger.info(f"Time taken: {end - start} seconds")

    app.logger.info(f"Audio duration: {audio_duration:.2f} seconds")
    app.logger.info(f"Real-time factor: {audio_duration / (end - start):.2f}")

    app.logger.info(genai_result)
    return str(genai_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0')