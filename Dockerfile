FROM python:3.7-slim

WORKDIR /home/auto_reports

RUN python -m venv venv
RUN . venv/bin/activate

RUN apt-get update
RUN apt-get install -y gcc && apt-get install -y g++

COPY oracle-client.deb oracle-client.deb
RUN apt-get install libaio1
RUN dpkg -i oracle-client.deb

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt; exit 0
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY app app
COPY reports reports
COPY meta meta
COPY static static
COPY templates templates
COPY uploaded_files uploaded_files
COPY downloaded_files downloaded_files
COPY config.py boot.sh ./

RUN chmod a+x boot.sh

ENTRYPOINT ["./boot.sh"]
