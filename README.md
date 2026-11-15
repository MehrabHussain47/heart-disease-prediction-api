# Heart Disease Prediction API

A deliberately simple FastAPI classifier deployment exercise. It predicts whether heart disease is present; it is **not medical advice**.

## 1. Prepare the dataset and train the model

1. Download `heart.csv` from the [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).
2. Create the expected folders and place the CSV at `data/heart.csv`.
3. Install dependencies and train:

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python train_model.py --data data/heart.csv

This writes `model/heart_model.joblib`. Commit that generated file to the GitHub repository so Render can build the image without retraining.

## 2. Run locally with Docker

docker-compose build
docker-compose up

Open Swagger UI at http://localhost:8000/docs.

Quick test:

curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

Other endpoints:

curl http://localhost:8000/health
curl http://localhost:8000/info

## 3. Deploy to Render

1. Push this project **including `model/heart_model.joblib`** to GitHub.
2. In Render, choose **New + → Web Service** and connect the repository.
3. Select **Docker** as the environment. Keep the build context at the repository root because the Dockerfile is there.
4. Create the service and wait for the build to finish. Render provides the `PORT` environment variable; the Docker command uses it automatically.
5. Test `https://YOUR-SERVICE.onrender.com/health`, `/info`, and `/docs`.

For a deployed prediction:

curl -X POST https://YOUR-SERVICE.onrender.com/predict \\
  -H "Content-Type: application/json" \\
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

## Project layout

.
├── Dockerfile
├── docker-compose.yml
├── main.py
├── schemas.py
├── train_model.py
├── requirements.txt
├── data/heart.csv                 # downloaded from Kaggle; optional to commit
└── model/heart_model.joblib       # generated; commit for deployment