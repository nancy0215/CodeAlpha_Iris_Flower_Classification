import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv('iris.csv')
X = df.drop('species', axis=1)
y = df['species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training on {len(X_train)} flowers, testing on {len(X_test)} flowers")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=le.classes_, columns=le.classes_))

joblib.dump(model, 'iris_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Model saved as iris_model.pkl")
