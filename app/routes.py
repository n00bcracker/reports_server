import os
from flask import flash, request, redirect, url_for, render_template, send_from_directory
from sqlalchemy.exc import DatabaseError
from werkzeug.utils import secure_filename
from transliterate import translit
from reports.portf_report import NotValidClients, make_portf_cmp_report

from app import app
ALLOWED_EXTENSIONS = {'xls', 'xlsx',}

@app.route('/portfolio_cmp_report', methods=["GET", "POST"])
def upload_clients_sample():
    proccesed_file = None
    errors = dict()
    errors['request_file_error'] = None
    errors['other_file_error'] = None
    if request.method == 'GET':
        return render_template('clients_upload.html', proccesed_file=proccesed_file, errors=errors)
    if request.method == 'POST':
        file = request.files['requested_cl_file']
        if file.filename == '':
            errors['request_file_error'] = 'Файл не выбран.'
        elif request.content_length > 5 * 1024 * 1024:
            errors['request_file_error'] = 'Размер файла превышает 5 Мб.'
        else:
            filename_parts = file.filename.rsplit('.', 1)
            file_ext = filename_parts[1].lower() if len(filename_parts) == 2 else ''
            if file_ext not in ALLOWED_EXTENSIONS:
                errors['request_file_error'] = 'Выбран не Excel-файл.'
            else:
                filename = translit(file.filename, 'ru', reversed=True)
                filename = secure_filename(filename)
                rq_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(rq_filename)

                sign_level = request.form.get('sign_level')

                try:
                    group = request.form.get('cmp_group')
                    if group == 'active':
                        proccesed_file = make_portf_cmp_report(rq_filename, sign_level=sign_level, only_active=True)
                    elif group == 'all':
                        proccesed_file = make_portf_cmp_report(rq_filename, sign_level=sign_level)
                    elif group == 'other':
                        file = request.files['other_cl_file']
                        if file.filename == '':
                            errors['other_file_error'] = 'Файл не выбран.'
                        else:
                            filename_parts = file.filename.rsplit('.', 1)
                            file_ext = filename_parts[1].lower() if len(filename_parts) == 2 else ''
                            if file_ext not in ALLOWED_EXTENSIONS:
                                errors['other_file_error'] = 'Выбран не Excel-файл.'
                            else:
                                filename = translit(file.filename, 'ru', reversed=True)
                                filename = secure_filename(filename)
                                oth_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                                file.save(oth_filename)

                                proccesed_file = make_portf_cmp_report(rq_filename, sign_level=sign_level,\
                                                                        other_clients_filename=oth_filename)
                except NotValidClients as e:
                    if e.file_id == 'requested_cl_file':
                        errors['request_file_error'] = e.message
                    elif e.file_id == 'other_cl_file':
                        errors['other_cl_file'] = e.message
                except DatabaseError:
                    errors['request_file_error'] = 'В данный момент имеются проблемы с базой данных. Попробуйте позднее.'

        return render_template('clients_upload.html', proccesed_file=proccesed_file, errors=errors)

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