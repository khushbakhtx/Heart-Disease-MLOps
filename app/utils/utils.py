import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc


def validate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series):

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, preds))

    return accuracy


def plot_feature_importance(model: object):

    feature_importance = model.get_feature_importance()
    feature_names = model.feature_names_
    sorted_idx = np.argsort(feature_importance)

    fig = plt.figure(figsize=(8, 4))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.title('Catboost feature importance')
    plt.show()

    return fig

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)[:, 1] 

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()

    return plt.gcf()

