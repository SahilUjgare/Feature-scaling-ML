# 📘 Feature Scaling in Machine Learning

Feature scaling is an essential preprocessing step in Machine Learning that brings all numerical features to a similar scale. Many ML algorithms perform better or converge faster when features are scaled properly.

---

## 🚀 Why Feature Scaling?

Different features in datasets often have different ranges.
Example:

* **Age:** 18–80
* **Salary:** 20,000–200,000

Models like Gradient Descent–based algorithms, KNN, SVM, and Neural Networks are sensitive to such differences. Feature scaling solves this by normalizing or standardizing the values.

---

## 🔧 Types of Feature Scaling

### 1️⃣ **Min-Max Scaling (Normalization)**

Rescales features to a **fixed range**, usually 0 to 1.

**Formula:**

```
X_scaled = (X - X_min) / (X_max - X_min)
```

**When to use:**
✔ Gradient descent algorithms
✔ Neural networks
✔ Distance-based models like KNN

---

### 2️⃣ **Standardization (Z-score Normalization)**

Transforms data to have **mean = 0** and **std deviation = 1**.

**Formula:**

```
X_scaled = (X - mean) / std
```

**When to use:**
✔ Works well with most algorithms
✔ Useful when your data has outliers

---

### 3️⃣ **Robust Scaling**

Handles extreme outliers better by using median and IQR.

**Formula:**

```
X_scaled = (X - median) / IQR
```

---

## 🛠 Example (Python)

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler1 = MinMaxScaler()
scaled_data_1 = scaler1.fit_transform(data)

scaler2 = StandardScaler()
scaled_data_2 = scaler2.fit_transform(data)
```

---

## 📊 Algorithms That Need Scaling

* K-Nearest Neighbors (KNN)
* Support Vector Machines (SVM)
* Logistic Regression
* Linear / Ridge / Lasso Regression
* Neural Networks
* Gradient Descent–based models

---

## 📌 Algorithms *Not* Sensitive to Scaling

* Decision Trees
* Random Forest
* XGBoost
