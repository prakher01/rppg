from flask import Flask
from routes import setup_routes

app = Flask(__name__)

# Set up temporary directory
import os
import tempfile
TEMP_DIR = tempfile.gettempdir()
os.makedirs('static', exist_ok=True)

setup_routes(app)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)