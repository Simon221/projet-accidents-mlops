
# Run locally (no docker)
1. Create a virtualenv and install requirements: `pip install -r requirements.txt`
2. start mlflow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow/artifacts -p 5001`
3. Run the notebook or `python src/training/train.py --x-train data/X_train.csv --y-train data/y_train.csv --x-test data/X_test.csv --y-test data/y_test.csv --output-dir outputs --mlflow-uri http://127.0.0.1:5001`
4. Run Api : `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`
5. Run Prometheus
5. Run Grafana


# Run with docker-compose
1. Build and start : `docker-compose up --build` ou `docker compose up --build`
2. In another terminal, trigger trainer: `docker-compose run --rm trainer`
3. API available at http://localhost:8000 (health, predict endpoints)
3.1 Call des apis : Voir fichier request.txt
4. MLflow UI at http://localhost:5001
5. Prometheus at http://localhost:9090, Grafana at http://localhost:3000

# Livrable
1. Build l'image docker : `docker build -t projet_accidents -f Dockerfile .`
2. Exporter en tar : `docker save -o mon-projet_accidents.tar projet_accidents:latest`
3. Load ailleur : `docker load -i mon-projet_accidents.tar`

