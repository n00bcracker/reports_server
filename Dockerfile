FROM python:3.8-slim

WORKDIR /home/auto_reports

COPY oracle-client.deb oracle-client.deb

RUN apt-get update && \
    apt-get install -y libaio1 && \
    dpkg -i oracle-client.deb && rm oracle-client.deb && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY uploaded_files uploaded_files
COPY downloaded_files downloaded_files
COPY app app
COPY reports reports
COPY meta meta
COPY static static
COPY templates templates
COPY config.py boot.sh ./

RUN chmod a+x boot.sh

ENTRYPOINT ["./boot.sh"]
