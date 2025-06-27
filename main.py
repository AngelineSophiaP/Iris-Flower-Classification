from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
  

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())


x = df.drop('target', axis=1)
y = df['target']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)


acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.2f}")

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))
