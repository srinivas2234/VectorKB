# syntax=docker/dockerfile:1
FROM python:3.11.5
WORKDIR /app
COPY . /app
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 5000
CMD [ "python", "app.py"]