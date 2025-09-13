# 🚦 Accidents MLOps Project (Datascientest)

Ce projet illustre un **pipeline MLOps minimal** intégrant :  
- **Entraînement** d’un modèle ML (RandomForest)  
- **Suivi des expériences** avec **MLflow**  
- **Espaces réservés pour DVC** (versionnage des données et pipelines)  
- **API d’inférence** via **FastAPI**  
- **Supervision** avec **Prometheus / Grafana**  
- **Orchestration** via **Docker Compose**  

---

## 🚀 Démarrer le projet avec Docker Compose

### 0. Déplaces toi dans l'environnement virtuel (mlops_env) et active le 
```bash
cd accidents-mlops
# et
source mlops_env/bin/activate
```

### 1. Build & Start
```bash
docker-compose up --build
# ou
docker compose up --build
```

### 2. Lancer l’entraînement
Dans un autre terminal :
```bash
docker-compose run --rm trainer
```
Le modèle sera entraîné et sauvegardé dans `outputs/model.pkl`.

---

## 📊 MLflow UI
Accédez à l’interface MLflow :  
👉 [http://localhost:5001](http://localhost:5001)  

Vous pourrez y consulter les expériences, métriques et artefacts des modèles.

---

## 🌐 API FastAPI
L’API est accessible à :  
👉 [http://localhost:8000](http://localhost:8000)

Endpoints disponibles :
- `/admin/register` → endpoint protégé par un token admin
- `/health` → check de l’état du service  
- `/predict` → effectuer une prédiction  
  

### 🔧 Exemples de requêtes (voir aussi `request.txt`)

#### ✅ GET Health
```bash
curl -X GET http://localhost:8000/health
```

#### ✅ POST admin/register (OK)
```bash
curl -X POST http://localhost:8000/admin/register   -H "x-admin-token: change-me-secure-token"   -H "Content-Type: application/json"   -d '{"action":"test"}'
```

#### ❌ POST admin/register (Forbidden)
```bash
curl -X POST http://localhost:8000/admin/register   -H "x-admin-token: another-secure-token"   -H "Content-Type: application/json"   -d '{"action":"test"}'
```

#### ✅ POST Predict
```bash
curl -X POST http://localhost:8000/predict   -H "Content-Type: application/json"   -d '{
        "features": {
          "id_usager": "203813884",
          "place": 2,
          "catu": 2,
          "sexe": 2,
          "secu1": 1.0,
          "year_acc": 2023,
          "victim_age": 46.0,
          "catv": 2.0,
          "obsm": 2.0,
          "motor": 1.0,
          "catr": 3,
          "circ": 2.0,
          "surf": 2.0,
          "situ": 1.0,
          "vma": 90.0,
          "jour": 25,
          "mois": 4,
          "lum": 2,
          "dep": 34,
          "com": 34333,
          "agg_": 1,
          "int": 1,
          "atm": 1.0,
          "col": 5.0,
          "lat": 43.475221,
          "long": 3.770757,
          "hour": 21,
          "nb_victim": 6,
          "nb_vehicules": 3
        }
      }'
```

---

## 📈 Monitoring avec Prometheus & Grafana

### 🔹 Prometheus
👉 [http://localhost:9090](http://localhost:9090)  
Permet de consulter directement les métriques exposées par l’API.

### 🔹 Grafana
👉 [http://localhost:3000](http://localhost:3000)  
Login par défaut : **admin / admin**  

#### Importer le Dashboard
1. Aller dans **Dashboards > Import**  
2. Importer le JSON situé dans `grafana/dashboard.json`  
3. Associer la datasource **Prometheus**  
4. Sauvegarder   

> ℹ️ **Note** :  
Si vous voyez le message *"Datasource Prometheus was not found"*, créez une datasource manuellement :  
- Aller dans **Connections > Data sources > Add data source**  
- Choisir **Prometheus**  
- Mettre l’URL : `http://prometheus:9090`  
- Donner le nom **Prometheus**  
- Enregistrer  


## 🚀 Démarrer le projet en local
### 0. Déplaces toi dans l'environnement virtuel (mlops_env) et active le. A defaut installer le requirements.txt
```bash
cd accidents-mlops
# et
source mlops_env/bin/activate

```
### 1. Démarrer le MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow/artifacts -p 5001
```

### 2. Lancer un Training
```bash
python src/training/train.py --x-train data/X_train.csv --y-train data/y_train.csv --x-test data/X_test.csv --y-test data/y_test.csv --output-dir outputs --mlflow-uri http://127.0.0.1:5001
```

### 3. Lancer l'API
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
### 🔧 Exemples de requêtes (voir aussi `request.txt`)

#### ✅ GET Health
```bash
curl -X GET http://localhost:8000/health
```

#### ✅ POST admin/register (OK)
```bash
curl -X POST http://localhost:8000/admin/register   -H "x-admin-token: change-me-secure-token"   -H "Content-Type: application/json"   -d '{"action":"test"}'
```
