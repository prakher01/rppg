from flask import render_template, request, Response, send_from_directory, jsonify
import os
import uuid
import json
from processing import process_video

def setup_routes(app):
    @app.route('/')
    def index():
        return render_template('heart_video.html')

    @app.route('/upload', methods=['POST'])
    def upload():
        video_file = request.files['video']
        temp_filename = f"{uuid.uuid4()}.mp4"
        temp_path = os.path.join(app.config['TEMP_DIR'], temp_filename)
        video_file.save(temp_path)
        return Response(process_video(temp_path), mimetype='text/event-stream')

    @app.route('/static/<filename>')
    def serve_static(filename):
        return send_from_directory('static', filename)