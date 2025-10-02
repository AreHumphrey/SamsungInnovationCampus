import pandas as pd
import time
from sklearn.ensemble import GradientBoostingRegressor

print("загрузка тренировочных данных")
train = pd.read_csv('train.csv')
print(f"Загружено {train.shape[0]} строк и {train.shape[1]} признаков")

print("загрузка тестовых данных")
test = pd.read_csv('test.csv')
print(f"Загружено {test.shape[0]} строк для предсказания")

print("подготовка признаков и целевой переменной")
X_train = train.drop('critical_temp', axis=1)
y_train = train['critical_temp']
print(f"Целевая переменная: critical_temp (диапазон: {y_train.min():.1f} – {y_train.max():.1f} K)")

print("Начало обучения модели")
start_time = time.time()
model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42,
    verbose=1
)
model.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Модель обучена за {train_time:.1f} секунд")

print("предсказания на тестовом наборе")
predictions = model.predict(test)
print(f"Первые 5 предсказаний: {[round(p, 2) for p in predictions[:5]]}")

print("сохранение результата в файл")
pd.DataFrame({'critical_temp': predictions}).to_csv('predict.csv', index=False)
print("сохранен")