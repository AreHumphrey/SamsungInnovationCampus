import pandas as pd
from catboost import CatBoostClassifier

# Загрузка данных
train = pd.read_csv('train_oil.csv')
test = pd.read_csv('oil_test.csv')

target_col = 'Onshore/Offshore'
features_to_drop = ['Field name', target_col]

# Определяем категориальные признаки (object-типы)
X_temp = train.drop(columns=['Field name'])
categorical_features = [
    col for col in X_temp.columns
    if col != target_col and X_temp[col].dtype == 'object'
]

# Подготавливаем X_train и X_test
X_train = train.drop(columns=features_to_drop)
y_train = train[target_col]

X_test = test.drop(columns=['Field name'])

# Заполняем NaN в категориальных колонках значением 'MISSING'
for col in categorical_features:
    X_train[col] = X_train[col].fillna('MISSING').astype(str)
    X_test[col] = X_test[col].fillna('MISSING').astype(str)

# Также убедимся, что числовые колонки не содержат NaN (хотя CatBoost с ними справляется лучше)
# Но для надёжности можно заполнить их средним (опционально)
numeric_features = X_train.select_dtypes(include=['number']).columns
X_train[numeric_features] = X_train[numeric_features].fillna(X_train[numeric_features].mean())
X_test[numeric_features] = X_test[numeric_features].fillna(X_train[numeric_features].mean())  # используем среднее из train!

# Обучение модели
model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=0
)

model.fit(X_train, y_train, cat_features=categorical_features)


predictions = model.predict(X_test)

# Убедимся, что predictions — одномерный массив
if predictions.ndim > 1:
    predictions = predictions.flatten()

# Сохранение
output = pd.DataFrame({
    'Index': test.index,
    'Onshore/Offshore': predictions
})

output.to_csv('predict.csv', index=False)