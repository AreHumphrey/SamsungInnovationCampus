import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


train = pd.read_csv('train_final.csv')
print(f"Загружено {train.shape[0]} строк и {train.shape[1]} признаков")

test = pd.read_csv('test_final.csv')
print(f"Загружено {test.shape[0]} строк для предсказания")

print("\nКолонки в тренировочном файле:", train.columns.tolist())

target_col = 'is_canceled'
if target_col not in train.columns:
    raise KeyError(f"Колонка '{target_col}' отсутствует в train.csv")

y_train = train[target_col]
X_train = train.drop(columns=[target_col])
print(f"Целевая переменная: {target_col} (0/1), уникальных значений: {y_train.nunique()}")

def preprocess_data(df):
    df = df.copy()

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna('Unknown', inplace=True)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


X_train_processed = preprocess_data(X_train)
X_test_processed = preprocess_data(test)


X_tr, X_val, y_tr, y_val = train_test_split(X_train_processed, y_train, test_size=0.2, random_state=42, stratify=y_train)


model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
model.fit(X_tr, y_tr)


y_val_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_val_pred)
print(f"Accuracy на валидации: {acc:.4f}")

y_pred = model.predict(X_test_processed)
print(f"Предсказания получены: {y_pred[:10]}... (первые 10 значений)")

assert set(y_pred).issubset({0, 1}), "Предсказания содержат недопустимые значения"

pd.DataFrame({'is_canceled': y_pred}).to_csv('predict.csv', index=False)
