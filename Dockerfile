FROM python:3.6.4

WORKDIR deepcolor
COPY . .
RUN pip install -r requirements.txt
CMD python app.py


