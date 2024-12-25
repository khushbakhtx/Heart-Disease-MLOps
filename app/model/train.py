from catboost import CatBoostClassifier

def train_model(X, y):

    model = CatBoostClassifier(
        depth = 6,
        iterations = 100,
        learning_rate = 0.1
    )

    model.fit(X, y)

    return model


