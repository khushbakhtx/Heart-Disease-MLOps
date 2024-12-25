import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(data: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):

    data.columns = data.columns.str.strip().str.lower()
    data['sex'] = (data['sex'] == 'M').astype(int)
    data['exerciseangina'] = (data['exerciseangina'] == 'Y').astype(int)

    chest_pain = pd.get_dummies(data['chestpaintype'], prefix="chest_pain")
    resting_ecg = pd.get_dummies(data['restingecg'], prefix="restingecg")
    st_slope = pd.get_dummies(data['st_slope'], prefix="st_slope")

    data = data.drop(["chestpaintype", "restingecg", "st_slope"], axis=1)

    data = pd.concat([data, chest_pain, resting_ecg, st_slope], axis=1)

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    data.fillna(data.mean(), inplace=True)

    X = data.drop(["heartdisease"], axis=1)
    y = data["heartdisease"]

    X = calculate_features(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)

    print(X_train.shape, X_test.shape)

    return X_train, X_test, y_train, y_test


def calculate_features(data: pd.DataFrame):
    data['maxhr_squared'] = data['maxhr'] ** 2
    data['cholesterol_squared'] = data['cholesterol'] ** 2
    return data
