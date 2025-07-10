
from flask import Flask, render_template
from video_routes import video_blueprint
import os
import tempfile

app = Flask(__name__)

# Register Blueprint
app.register_blueprint(video_blueprint)

# Configuration
app.config['TEMP_DIR'] = tempfile.gettempdir()
os.makedirs('static', exist_ok=True)

# Root Route
@app.route('/')
def index():
    return render_template('heart_video.html')

# Run the App
if __name__ == '__main__':
    app.run(debug=True, threaded=True)

