
# 🌸 Iris Flower Classification using K-Nearest Neighbors (KNN)

## 📌 Project Description

This project aims to classify **Iris flowers** into one of three species:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

The classification is based on four features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

We use the **K-Nearest Neighbors (KNN)** algorithm to group each input into its most 
likely flower species based on feature similarity.

---

## 🎯 Objective

To build a machine learning model using KNN that can accurately classify the type of Iris
flower based on input measurements.

---

## 🗂️ Dataset

- **Source:** `sklearn.datasets.load_iris()`
- **Total Samples:** 150
- **Features:** 4 numerical attributes
- **Target Classes:** 3 (Setosa, Versicolor, Virginica)

---

## ⚙️ Libraries Used

- `pandas` — data manipulation
- `sklearn.datasets` — to load the Iris dataset
- `sklearn.model_selection` — train/test split
- `sklearn.neighbors` — KNN Classifier
- `sklearn.metrics` — for evaluation (accuracy, confusion matrix, classification report)

---

## 🧠 Algorithm Used: K-Nearest Neighbors (KNN)

- KNN is a **supervised classification** algorithm.
- It classifies new data points based on the **majority vote** of the k-nearest points in the training set.
- In this project, we used **k = 3** (i.e., `n_neighbors=3`).

---

## 🛠️ ML Workflow

1. **Load** the Iris dataset using `sklearn`.
2. **Create** a DataFrame and include the target column.
3. **Split** the data into training and testing sets.
4. **Train** a KNN classifier using the training data.
5. **Predict** the flower type on the test data.
6. **Evaluate** the model using:
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

---

## 📈 Results

```text
✅ Accuracy: 1.00

📊 Confusion Matrix:
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]

📄 Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         9
           2       1.00      1.00      1.00        11
````

---

## ✅ Conclusion

The KNN model achieved **100% accuracy** on the Iris dataset, demonstrating that KNN is highly effective for this
type of classification task, especially when the dataset is small and well-separated.

---

## 📁 Folder Structure (if applicable)

```
├── iris_knn_classification.ipynb  # or .py file
├── README.md
```

