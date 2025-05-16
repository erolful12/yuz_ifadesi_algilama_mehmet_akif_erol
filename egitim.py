import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

veri = pd.read_csv('veriseti.csv')
X = veri.drop('etiket', axis=1)
y = veri['etiket']

X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_egitim, y_egitim)

y_tahmin = model.predict(X_test)
dogruluk = accuracy_score(y_test, y_tahmin)
print(f"Model doğruluk oranı: {dogruluk:.2f}")

joblib.dump(model, 'model.pkl')
print("Model kaydedildi: model.pkl")