import os
import tempfile
import logging
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import gigaam
from gigaam.model import LONGFORM_THRESHOLD

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# Load available models
AVAILABLE_MODELS = [
    'ssl', 'v1_ssl',
    'ctc', 'v2_ctc', 'v1_ctc',
    "rnnt", "v2_rnnt", "v1_rnnt",
]

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25MB max
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/models', methods=['GET'])
def models():
    result = {"object": "list", "data": AVAILABLE_MODELS}
    return jsonify(result), 200, {'Content-Type': 'application/json'}


@app.route('/models/<model>', methods=['GET'])
def model(model_id: str):
    if model_id in AVAILABLE_MODELS:
        result = {"id": model_id, "object": "model", "owned_by": "gigaam"}
        return jsonify(result), 200, {'Content-Type': 'application/json'}
    else:
        return jsonify({'message': 'Model not found'}), 404


def format_time_srt(seconds: float) -> str:
    """
    Convert seconds to SRT time format (HH:MM:SS,mmm).
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"


def format_time_vtt(seconds: float) -> str:
    """
    Convert seconds to WebVTT time format (HH:MM:SS.mmm).
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02}.{milliseconds:03}"


@torch.inference_mode()
def transcribe(model, wav_file: str) -> list:
    """
    Transcribes a short audio file into text.
    """
    wav, length = model.prepare_wav(wav_file)
    if length > LONGFORM_THRESHOLD:
        raise ValueError("Too long wav file, use 'transcribe_longform' method.")

    encoded, encoded_len = model.forward(wav, length)
    result = model.decoding.decode(model.head, encoded, encoded_len)[0]
    return [{
        "transcription": result,
        "boundaries": [0, encoded_len.item()],
    }]


@app.route('/audio/transcriptions', methods=['POST'])
def transcriptions():
    _log.info('Received a POST request to transcribe an audio file.')

    # Check if 'file' attribute is present in the request
    if 'file' not in request.files:
        return jsonify({'message': 'No file attribute provided in the request'}), 400

    file = request.files['file']
    if file.filename == '' or file is None:
        return jsonify({'message': 'File attribute is empty'}), 400

    # Check if model parameter is provided and validate it
    if 'model' not in request.form:
        return jsonify({'message': 'No model provided in the request'}), 400

    model_id = request.form['model']
    if model_id not in AVAILABLE_MODELS:
        return jsonify({'message': f'Requested model "{model_id}" is not available.'}), 400

    # Collect form data for transcription
    data = {}
    if 'use_flash' in request.form:
        data['use_flash'] = request.form['use_flash']
    if 'use_longform' in request.form:
        data['use_longform'] = request.form['use_longform']
    if 'response_format' in request.form:
        data['response_format'] = request.form['response_format']
    if 'temperature' in request.form:
        data['temperature'] = float(request.form['temperature'])

    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name
        file.save(temp_file_path)

    try:
        # Flash attention
        use_flash = bool(data.get('use_flash', False))

        # Load the GigaAM model
        giga_model = gigaam.load_model(model_id, use_flash=use_flash)

        # Process the audio file
        use_longform = data.get('use_longform', False)
        if use_longform:
            transcribed_segments = giga_model.transcribe_longform(temp_file_path, **data)
        else:
            transcribed_segments = transcribe(giga_model, temp_file_path, **data)

        # Determine the response type
        response_format = data.get('response_format', 'text')

        # Prepare the response based on the response type
        if response_format == 'json':
            response_data = {
                "segments": transcribed_segments,
                "model": model_id,
                "object": "transcription"
            }
            return jsonify(response_data), 200, {'Content-Type': 'application/json'}

        elif response_format == 'text':
            response_text = " ".join([segment["transcription"] for segment in transcribed_segments]).strip()
            return response_text, 200, {'Content-Type': 'text/plain'}

        elif response_format == 'srt':
            srt_content = ""
            for i, segment in enumerate(transcribed_segments, start=1):
                start_time, end_time = segment["boundaries"]
                transcription = segment["transcription"]
                srt_content += f"{i}\n"
                srt_content += f"{format_time_srt(start_time)} --> {format_time_srt(end_time)}\n"
                srt_content += f"{transcription}\n\n"
            return srt_content, 200, {'Content-Type': 'text/plain'}

        elif response_format == 'vtt':
            vtt_content = "WEBVTT\n\n"
            for i, segment in enumerate(transcribed_segments, start=1):
                start_time, end_time = segment["boundaries"]
                transcription = segment["transcription"]
                vtt_content += f"{i}\n"
                vtt_content += f"{format_time_vtt(start_time)} --> {format_time_vtt(end_time)}\n"
                vtt_content += f"{transcription}\n\n"
            return vtt_content, 200, {'Content-Type': 'text/plain'}

        else:
            return jsonify({'message': 'Invalid response_format specified.'}), 400

    except Exception as e:
        _log.error(f'An error occurred: {e}')
        return jsonify({'message': 'An error occurred while processing the audio file.'}), 500
    finally:
        # Remove the temporary file
        if 'temp_file_path' in locals():
            os.remove(temp_file_path)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
