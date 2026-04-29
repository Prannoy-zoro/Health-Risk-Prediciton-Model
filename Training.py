import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
#20240802468
df = pd.read_csv("cleaned.csv")


encoder=LabelEncoder()
df["NObeyesdad"] = encoder.fit_transform(df["NObeyesdad"])
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
#20240802468
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("Accuracy:", acc)
#20240802468
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
joblib.dump(acc,"accuracy.pkl")
joblib.dump(encoder,"encoder.pkl")