
FROM python:3.10-slim
WORKDIR /app
COPY src/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY src/api /app
ENV MODEL_PATH=/app/model/model.pkl
EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
