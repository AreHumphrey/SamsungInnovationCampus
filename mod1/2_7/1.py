import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Загрузка данных
print("🔍 Загружаем данные из файла Physical_Activity_Monitoring_unlabeled.csv...")
df = pd.read_csv('Physical_Activity_Monitoring_unlabeled.csv')

# Убираем ненужные столбцы
features = df.drop(columns=['timestamp', 'subject_id'], errors='ignore')

# Заполняем пропущенные значения
print("⚙️ Заполняем пропуски и масштабируем признаки...")
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)

# Попробуем разное количество кластеров (от 5 до 10) и выберем лучшее по силуэту
best_score = -1
best_n_clusters = 6
best_labels = None

print("🔍 Подбираем оптимальное количество кластеров...")
for n in range(5, 11):
    kmeans = KMeans(n_clusters=n, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(features_scaled)
    score = silhouette_score(features_scaled, labels)
    print(f"  n_clusters = {n}, silhouette = {score:.4f}")
    if score > best_score:
        best_score = score
        best_n_clusters = n
        best_labels = labels

print(f"✅ Лучшее количество кластеров: {best_n_clusters}, silhouette = {best_score:.4f}")

# Используем Gaussian Mixture для лучшего кластера
print("🧠 Обучаем GaussianMixture...")
gmm = GaussianMixture(n_components=best_n_clusters, random_state=42)
gmm_labels = gmm.fit_predict(features_scaled)

# Преобразование кластеров к порядку: 0->1, 1->2, ..., N-1->N
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
gmm_labels = le.fit_transform(gmm_labels) + 1

# Создание результата
print("💾 Создаём файл predict.csv...")
result = pd.DataFrame({
    'Index': range(1, len(gmm_labels) + 1),
    'activityID': gmm_labels
})

result.to_csv('predict.csv', index=False)
print("✅ Файл predict.csv успешно создан!")
print(f"📊 Пример первых 10 строк:\n{result.head(10)}")