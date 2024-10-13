import os
import subprocess
import tempfile
import logging
import uuid
import json
import argparse
from dotenv import load_dotenv
from google.cloud import storage
from flask import Flask, request, jsonify
from flask_cors import CORS

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RIVA_SERVER = os.getenv('RIVA_SERVER')
RIVA_FUNCTION_ID = os.getenv('RIVA_FUNCTION_ID')
RIVA_API_KEY = os.getenv('RIVA_API_KEY')
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')

app = Flask(__name__)
CORS(app)

def run_riva_asr(input_file):
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=RIVA_SERVER)
    parser.add_argument("--use-ssl", action="store_true")
    parser.add_argument("--metadata", nargs=2, action="append")
    parser.add_argument("--language-code", default='en-US')
    parser.add_argument("--input-file")

    args_list = [
        "--server", RIVA_SERVER,
        "--use-ssl",
        "--metadata", "function-id", RIVA_FUNCTION_ID,
        "--metadata", "authorization", f"Bearer {RIVA_API_KEY}",
        "--language-code", 'en-US',
        "--input-file", input_file
    ]

    args, unknown_args = parser.parse_known_args(args_list)

    command = ["python", "transcribe_file_offline.py"] + args_list
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        logger.error(f"Error output: {e.output}")
        raise

def save_text_to_gcs(text):
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    
    filename = f"{uuid.uuid4()}.txt"
    blob = bucket.blob(filename)
    blob.upload_from_string(text)
    
    return blob.public_url

def extract_transcript(asr_output):
    if not asr_output:
        raise ValueError("ASR output is empty. Check the run_riva_asr function.")
    
    lines = asr_output.strip().split('\n')
    
    for line in reversed(lines): 
        if line.startswith("Final transcript:"):
            return line.split("Final transcript:")[1].strip()
    
    transcript = ""
    for line in lines:
        if line.strip().startswith("transcript:"):
            transcript += line.split("transcript:")[1].strip().strip('"') + " "
    
    if transcript:
        return transcript.strip()
    
    raise ValueError("Unable to extract transcript from ASR output")

@app.route('/asr', methods=['POST'])
def asr_handler():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.split('.')[-1].lower() in ['wav', 'opus', 'ogg']:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            file.save(temp_file.name)
            input_file = temp_file.name
        
        try:
            asr_output = run_riva_asr(input_file)
            logger.info(f"ASR OUTPUT: {asr_output}")
            if not asr_output:
                raise ValueError("ASR output is empty")
            
            transcript = extract_transcript(asr_output)
            text_url = save_text_to_gcs(transcript)
            
            return jsonify({
                'text_url': text_url,
                'text': transcript
            })
        except ValueError as e:
            logger.error(f"Transcript extraction error: {str(e)}")
            return jsonify({'error': str(e)}), 500
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            return jsonify({'error': 'An unexpected error occurred during processing'}), 500
        finally:
            if os.path.exists(input_file):
                os.unlink(input_file)
    else:
        return jsonify({'error': 'Invalid file type. Please upload a WAV, OPUS, or OGG file.'}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))