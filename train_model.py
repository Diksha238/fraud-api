import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

data = pd.read_csv("creditcard.csv")
data = data.sample(frac=1, random_state=42)
data = data.sample(n=50000, random_state=42)

X = data.drop("Class", axis=1)
y = data["Class"]

# Step 1: Split first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Upsample ONLY train data
train_data = pd.concat([X_train, y_train], axis=1)
fraud = train_data[train_data.Class == 1]   # ✅ train_data
normal = train_data[train_data.Class == 0]  # ✅ train_data

fraud_upsampled = resample(
    fraud,
    replace=True,
    n_samples=len(normal),
    random_state=42
)

train_balanced = pd.concat([normal, fraud_upsampled])
X_train = train_balanced.drop("Class", axis=1)
y_train = train_balanced["Class"]

# n_estimators badha — better model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "fraud_model.pkl")