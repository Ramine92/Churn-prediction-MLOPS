FROM python:3.11-slim
WORKDIR /app
#install dependencies first (cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "dvc-s3" "s3fs<2025"

COPY . .
#pull models .pkl from daghubs remote repository
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
RUN dvc pull

EXPOSE 8000
CMD ["uvicorn", "app.main:app","--host","0.0.0.0","--port","8000"]
