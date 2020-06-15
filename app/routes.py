import os
from flask import flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
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
            if file.filename == '':
                error = 'Файл не выбран.'
            elif request.content_length > 5 * 1024 * 1024:
                error = 'Размер файла превышает 5 Мб.'
            else:
                filename_parts = file.filename.rsplit('.', 1)
                file_ext = filename_parts[1].lower() if len(filename_parts) == 2 else ''
                if file_ext not in ALLOWED_EXTENSIONS:
                    error = 'Выбран не Excel-файл.'
                else:
                    group = request.form.get('cmp_group')
                    only_active = True if group == 'active' else False
                    filename = translit(file.filename, 'ru', reversed=True)
                    filename = secure_filename(filename)
                    filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filename)
                    proccesed_file = make_portf_cmp_report(filename, only_active)

            return render_template('clients_upload.html', proccesed_file=proccesed_file, error=error)

@app.route('/downloaded_files/<path:filename>')
def download_file(filename):
    project_dir = os.getcwd()
    abs_dir = os.path.join(project_dir, app.config['DOWNLOAD_FOLDER'])
    resp = send_from_directory(abs_dir, filename, as_attachment=True)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers['Cache-Control'] = 'public, max-age=0'
    return resp

@app.route('/meta/<path:filename>')
def download_example(filename):
    project_dir = os.getcwd()
    abs_dir = os.path.join(project_dir, './meta')
    return send_from_directory(abs_dir, filename, as_attachment=True)