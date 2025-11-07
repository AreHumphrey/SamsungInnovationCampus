import pandas as pd
from catboost import CatBoostClassifier

train = pd.read_csv('train_oil.csv')
test = pd.read_csv('oil_test.csv')

target_col = 'Onshore/Offshore'
features_to_drop = ['Field name', target_col]

X_temp = train.drop(columns=['Field name'])
categorical_features = [
    col for col in X_temp.columns
    if col != target_col and X_temp[col].dtype == 'object'
]

X_train = train.drop(columns=features_to_drop)
y_train = train[target_col]

X_test = test.drop(columns=['Field name'])

for col in categorical_features:
    X_train[col] = X_train[col].fillna('MISSING').astype(str)
    X_test[col] = X_test[col].fillna('MISSING').astype(str)

numeric_features = X_train.select_dtypes(include=['number']).columns
X_train[numeric_features] = X_train[numeric_features].fillna(X_train[numeric_features].mean())
X_test[numeric_features] = X_test[numeric_features].fillna(X_train[numeric_features].mean()) 

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=0
)

model.fit(X_train, y_train, cat_features=categorical_features)


predictions = model.predict(X_test)

if predictions.ndim > 1:
    predictions = predictions.flatten()

output = pd.DataFrame({
    'Index': test.index,
    'Onshore/Offshore': predictions
})

output.to_csv('predict.csv', index=False)
