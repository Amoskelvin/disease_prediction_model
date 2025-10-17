import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

patients_data = pd.read_csv('medical_records.csv')

patients_data['sex'] = patients_data['sex'].map({'M': 0, 'F': 1})

X = patients_data.drop(columns=['label'])
y = patients_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
print('the accuracy of the prediction is', score)

joblib.dump(model, 'sickness_predict.joblib')
print('model has been saved as sickness_predict')