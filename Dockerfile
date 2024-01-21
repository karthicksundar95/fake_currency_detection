FROM continuumio/miniconda3:4.12.0
COPY . /usr/app
EXPOSE 8000
WORKDIR /usr/app
RUN pip3 install -r requirements.txt
CMD python swagger_app.py