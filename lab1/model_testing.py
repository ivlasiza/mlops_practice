from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import pickle

model = pickle.load(open('models/model.pkl', 'rb'))
df_test = pd.read_csv('test/test_result.csv')
X_test = df_test[['temp']]
y_test = df_test['is_noize']

predict = model.predict(X_test)

print("Metrics:")
print(f"Accuracy = {accuracy_score(y_test, predict)}")
print(f"Precision = {precision_score(y_test, predict)}")
print(f"Recall = {recall_score(y_test, predict)}")
print(f"F1-score = {f1_score(y_test, predict)}")

print("****** Pipiline finished ******")
