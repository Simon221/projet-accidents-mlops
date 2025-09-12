import os, argparse, joblib, mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def load_xy(x_path, y_path):
    X = pd.read_csv(x_path, index_col=None)
    y = pd.read_csv(y_path, index_col=None)

    # y → Série
    if y.shape[1] == 1:
        y = y.iloc[:, 0]

    # Nettoyer X
    for col in X.columns:
        if X[col].dtype == object:
            # supprimer espaces et caractères spéciaux
            X[col] = (
                X[col]
                .astype(str)
                .str.replace(r"[^\d\.\-]", "", regex=True)  # garder chiffres/.- seulement
            )
            # si convertible en numérique → cast
            X[col] = pd.to_numeric(X[col], errors="ignore")

    return X, y


def main(args):
    # tracking vers service Docker mlflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run() as run:
        X_train, y_train = load_xy(args.x_train, args.y_train)
        X_test, y_test = load_xy(args.x_test, args.y_test)

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        acc = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, proba) if proba is not None else np.nan
        except Exception:
            auc = np.nan

        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", float(acc))
        if not np.isnan(auc):
            mlflow.log_metric("roc_auc", float(auc))

        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "model.pkl")
        joblib.dump(clf, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        report_path = os.path.join(args.output_dir, "report.txt")
        with open(report_path, "w") as f:
            f.write(f"accuracy: {acc}\nauc: {auc}\n")
            f.write(str(classification_report(y_test, preds)))
        mlflow.log_artifact(report_path, artifact_path="reports")

        print("Training finished. Accuracy:", acc, " AUC:", auc)
        print("Run ID:", run.info.run_id)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--x-train", dest="x_train", default="/data/X_train.csv")
    p.add_argument("--y-train", dest="y_train", default="/data/y_train.csv")
    p.add_argument("--x-test", dest="x_test", default="/data/X_test.csv")
    p.add_argument("--y-test", dest="y_test", default="/data/y_test.csv")
    p.add_argument("--output-dir", default="/outputs")
    p.add_argument("--mlflow-uri", default="http://mlflow:5000")
    p.add_argument("--experiment-name", default="accidents_experiment")
    args = p.parse_args()
    main(args)
