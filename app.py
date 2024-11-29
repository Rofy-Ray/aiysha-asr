import os
import logging
import uuid
import torch
from pathlib import Path
import soundfile as sf
import librosa
import tempfile
from typing import Tuple
from dotenv import load_dotenv
from google.cloud import storage
from flask import Flask, request, jsonify
from flask_cors import CORS
import nemo.collections.asr as nemo_asr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')

app = Flask(__name__)
CORS(app)

class ASRProcessor:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """Lazy loading of the model to save memory until first request"""
        if self.model is None:
            logger.info("Loading ASR model...")
            self.model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                model_name="nvidia/parakeet-ctc-1.1b"
            )
            self.model = self.model.to(self.device)
            logger.info("ASR model loaded successfully")

    def process_audio(self, audio_path: str) -> str:
        """
        Process audio file and return transcription
        Returns: transcription
        # Returns: tuple of (transcription, duration_in_seconds)
        """
        audio, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        # duration = len(audio) / sample_rate

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            sf.write(temp_wav.name, audio, sample_rate, format='WAV')
            
            try:
                transcription = self.model.transcribe([temp_wav.name])[0]
                return transcription
            finally:
                os.unlink(temp_wav.name)

class GCSStorage:
    def __init__(self):
        self.client = storage.Client()
        self.bucket = self.client.bucket(GCS_BUCKET_NAME)

    def save_text(self, text: str) -> None:
        """Save text to GCS and return public URL"""
        filename = f"{uuid.uuid4()}.txt"
        blob = self.bucket.blob(filename)
        blob.upload_from_string(text)
        # return blob.public_url

asr_processor = ASRProcessor()
gcs_storage = GCSStorage()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/asr', methods=['POST'])
def asr_handler():
    """Handle ASR requests"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.wav'):
        return jsonify({'error': 'Invalid file type. Please upload a WAV file.'}), 400

    try:
        asr_processor.load_model()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            file.save(temp_file.name)
            
            try:
                transcript = asr_processor.process_audio(temp_file.name)
                
                gcs_storage.save_text(transcript)
                
                # text_url = gcs_storage.save_text(transcript)
                
                return jsonify({
                    'text': transcript
                })
            
            finally:
                os.unlink(temp_file.name)
                
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred during processing'}), 500

# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))