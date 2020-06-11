from flask import Flask
from config import UPLOAD_FOLDER, DOWNLOAD_FOLDER, FLASK_TEMPLATE_PATH

app = Flask(__name__, template_folder=FLASK_TEMPLATE_PATH)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # 100Mb

from app import routes