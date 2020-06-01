import os
from flask import flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from transliterate import translit
from reports.portf_report import make_portf_cmp_report

from app import app
ALLOWED_EXTENSIONS = {'xls', 'xlsx',}

@app.route('/portfolio_cmp_report', methods=["GET", "POST"])
def upload_clients_sample():
    proccesed_file = None
    error = None
    if request.method == 'GET':
        return render_template('clients_upload.html', proccesed_file=proccesed_file, error=error)
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        else:
            file = request.files['file']
            # if user does not select file, browser also submit an empty part without filename
            if file.filename == '':
                error = 'Файл не выбран.'
            else:
                filename_parts = file.filename.rsplit('.', 1)
                file_ext = filename_parts[1].lower() if len(filename_parts) == 2 else ''
                if file_ext not in ALLOWED_EXTENSIONS:
                    error = 'Выбран не Excel-файл.'
                else:
                    filename = translit(file.filename, 'ru', reversed=True)
                    filename = secure_filename(filename)
                    filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filename)
                    proccesed_file = make_portf_cmp_report(filename)

            return render_template('clients_upload.html', proccesed_file=proccesed_file, error=error)


@app.route('/downloads/<filename>')
def download_file(filename):
    send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)