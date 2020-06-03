import os

PROJECT_BASEDIR = os.path.dirname(os.path.abspath(__file__))
FLASK_TEMPLATE_PATH = os.path.join(PROJECT_BASEDIR, "templates")

UPLOAD_FOLDER = './uploaded_files'
DOWNLOAD_FOLDER = './downloaded_files'

ORACLE_USERNAME = os.getenv("ORACLE_USER") or None
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD") or None
ORACLE_TNS = os.getenv("ORACLE_TNS") or None

POTRFOLIO_TABLE = os.getenv("POTRFOLIO_TABLE") or None
