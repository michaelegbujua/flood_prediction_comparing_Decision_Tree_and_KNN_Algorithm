# 🌊 Flood Risk Prediction: Decision Tree vs K-Nearest Neighbors

## 📌 Project Overview

Flooding is one of the most disruptive environmental hazards, affecting communities, infrastructure, agriculture, transportation, and livelihoods. Data-driven prediction systems can help identify conditions associated with increased flood risk and support earlier decision-making.

This project explores **machine learning for flood-risk classification** by implementing and comparing two supervised learning algorithms:

* 🌳 **Decision Tree Classifier**
* 📍 **K-Nearest Neighbors (KNN) Classifier**

The primary objective is not simply to build a flood prediction model, but to **evaluate how two different classification approaches perform on the same flood-risk dataset** and determine how their predictive behaviour differs.

The target variable is **`Flood Risk`**, where:

* `0` = No Flood
* `1` = Flood

---

## 🎯 Project Objectives

The project was designed to:

1. Explore and understand the flood prediction dataset.
2. Perform basic exploratory data analysis (EDA).
3. Identify relationships between environmental variables and flood risk.
4. Transform categorical location data into machine-readable numerical features.
5. Build a **Decision Tree classification model**.
6. Build a **K-Nearest Neighbors classification model**.
7. Evaluate both models using classification performance measures and confusion matrices.
8. Compare their ability to correctly identify flood and non-flood conditions.
9. Examine which model provides stronger predictive performance on the available test data.

---

## 📊 Dataset

The dataset contains **10,000 observations** and the following variables:

| Feature             | Description                                  |
| ------------------- | -------------------------------------------- |
| `Date`              | Date associated with the observation         |
| `Location`          | City/location of the observation             |
| `Rainfall (mm)`     | Amount of rainfall                           |
| `Temperature (°C)`  | Recorded temperature                         |
| `Humidity (%)`      | Relative humidity                            |
| `River Level (m)`   | River/water level                            |
| `Soil Moisture (%)` | Soil moisture level                          |
| `Flood Risk`        | Target variable: `0` = No Flood, `1` = Flood |

### Dataset characteristics

The initial data inspection showed:

* **10,000 records**
* **8 columns**
* No missing values
* Five numerical environmental variables
* One categorical variable (`Location`)
* One date variable (`Date`)
* One binary classification target (`Flood Risk`)

The target contains two classes:

```text
0 → No Flood
1 → Flood
```

The dataset contains **1,139 flood-risk observations**, representing approximately **11.39%** of the dataset.

This class distribution is important because accuracy alone does not fully describe how effectively a model detects the relatively less frequent flood-risk class.

---

# 🔍 Exploratory Data Analysis

The notebooks perform several preliminary data-analysis steps, including:

* Dataset inspection
* Data-type inspection
* Descriptive statistics
* Correlation analysis
* Missing-value checks
* Target-variable inspection
* Date inspection

The correlation analysis indicates that **Rainfall** and **River Level** have the strongest observed linear relationships with `Flood Risk` among the numerical variables:

| Variable      | Correlation with Flood Risk |
| ------------- | --------------------------: |
| Rainfall      |                       0.409 |
| River Level   |                       0.420 |
| Temperature   |                       0.001 |
| Humidity      |                      -0.015 |
| Soil Moisture |                       0.007 |

This does not by itself establish causation or determine which variables a nonlinear model will consider most useful, but it provides useful exploratory insight into the dataset.

---

# Data Preprocessing

## Location Encoding

`Location` is categorical and therefore cannot be directly supplied to the machine-learning algorithms in its original text form.

The project converts locations into numerical representations using **one-hot encoding**.

The Decision Tree notebook creates dummy variables for:

```text
CityA
CityB
CityC
CityD
```

The KNN notebook similarly applies one-hot encoding using:

```python
pd.get_dummies(data, columns=['Location'], prefix='Loc')
```

## Date Handling

The `Date` column is removed before model training because it was not used as a predictive feature in the implemented models.

---

# 🌳 Model 1 — Decision Tree

The first approach uses Scikit-learn's:

```python
DecisionTreeClassifier(random_state=44)
```

### Training configuration

The data is divided into:

```text
Training set: 70%
Testing set: 30%
random_state: 44
```

The model is then trained using:

```python
model.fit(x_train, y_train)
```

and predictions are generated on the test set.

---

## Decision Tree Results

The Decision Tree achieved:

### Accuracy

```text
99.97%
```

More precisely:

```text
0.9996666666666667
```

### Confusion Matrix

```text
[[2643,    1],
 [   0,  356]]
```

Interpreting the matrix:

|                     | Predicted No Flood | Predicted Flood |
| ------------------- | -----------------: | --------------: |
| **Actual No Flood** |               2643 |               1 |
| **Actual Flood**    |                  0 |             356 |

This means that, on the Decision Tree test set:

* 2,643 non-flood observations were correctly classified.
* 356 flood observations were correctly classified.
* Only 1 non-flood observation was incorrectly classified as flood.
* No flood observations were incorrectly classified as non-flood.

The classification report also produced approximately perfect precision, recall and F1-score values when rounded to two decimal places.

---

# 📍 Model 2 — K-Nearest Neighbors

The second approach uses the **K-Nearest Neighbors (KNN)** algorithm.

The implemented classifier is:

```python
KNeighborsClassifier(
    n_neighbors=5,
    metric='minkowski',
    p=2
)
```

This configuration uses:

* **5 nearest neighbours**
* **Minkowski distance**
* `p=2`, corresponding to Euclidean distance

### Training configuration

The KNN notebook uses:

```text
Training set: 71%
Testing set: 29%
random_state: 0
```

The model then predicts the flood-risk class of the test observations.

---

# 📉 KNN Results

The KNN confusion matrix is:

```text
[[2473,   70],
 [ 201,  156]]
```

|                     | Predicted No Flood | Predicted Flood |
| ------------------- | -----------------: | --------------: |
| **Actual No Flood** |               2473 |              70 |
| **Actual Flood**    |                201 |             156 |

From this confusion matrix:

* 2,473 non-flood observations were correctly classified.
* 156 flood observations were correctly classified.
* 70 non-flood observations were incorrectly classified as flood.
* **201 actual flood observations were incorrectly classified as non-flood.**

The resulting accuracy is approximately:

```text
90.66%
```

The confusion matrix also reveals an important weakness: although the overall accuracy is above 90%, the model misses a considerable number of actual flood-risk cases.

---

# ⚔️ Model Comparison

The two models produced noticeably different results on their respective test sets.

| Metric                       | Decision Tree |        KNN |
| ---------------------------- | ------------: | ---------: |
| Algorithm                    | Decision Tree |        KNN |
| Test-set proportion          |           30% |        29% |
| Random state                 |            44 |          0 |
| Accuracy                     |    **99.97%** | **90.66%** |
| Correct No-Flood predictions |          2643 |       2473 |
| Incorrect No-Flood → Flood   |             1 |         70 |
| Correct Flood predictions    |       **356** |        156 |
| Incorrect Flood → No-Flood   |         **0** |    **201** |

### Accuracy comparison

```text
Decision Tree: 99.97%
KNN:           90.66%
```

Based on the results produced in these notebooks, the Decision Tree demonstrates substantially higher predictive performance on its test set.

However, the most important observation is not just the difference in accuracy.

### 🚨 Flood detection matters

For a flood-risk classification problem, failing to identify an actual flood condition can be particularly important.

The Decision Tree produced:

```text
False Negatives = 0
```

while KNN produced:

```text
False Negatives = 201
```

In other words, the Decision Tree test results show that it correctly identified all 356 flood-risk observations in its test set, while KNN failed to identify 201 of its 357 actual flood-risk observations.

This makes the confusion matrix an especially important part of the comparison rather than relying on accuracy alone.

---

# 🧠 What This Comparison Demonstrates

The project illustrates an important machine-learning principle:

> **A model should not be evaluated using accuracy alone.**

For classification problems, especially where one class represents a potentially important event such as flooding, it is useful to examine:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

A model can achieve a relatively high overall accuracy while still performing poorly at identifying the minority class.

The KNN results demonstrate this clearly: its approximately **90.66% accuracy** does not fully communicate the fact that **201 flood-risk observations were classified as non-flood**.

---

# 🔬 Key Technical Observation

The KNN notebook imports:

```python
from sklearn.preprocessing import StandardScaler
```

but the implemented workflow does not actually apply feature scaling before fitting KNN.

This is an important consideration because KNN is a **distance-based algorithm**. The numerical variables have different scales and ranges, for example:

* Rainfall: approximately 0–300 mm
* Temperature: approximately 10–35°C
* Humidity: approximately 30–100%
* River Level: approximately 1–10 m
* Soil Moisture: approximately 5–50%

A more rigorous KNN experiment would therefore standardize the features before calculating distances.

---

# ⚠️ Important Comparison Limitation

Although both notebooks use the same underlying dataset, the experiments do **not use identical train/test configurations**.

### Decision Tree

```python
test_size=0.3
random_state=44
```

### KNN

```python
test_size=0.29
random_state=0
```

Therefore, the reported results should be interpreted as the performance obtained in each notebook rather than as a perfectly controlled scientific benchmark.

For a stronger model-to-model comparison, both algorithms should ideally use:

* The same train/test split
* The same random state
* The same preprocessing pipeline
* The same feature set
* Appropriate feature scaling for KNN
* The same evaluation metrics
* Preferably cross-validation

This would make the comparison more statistically consistent.

---

# 🏁 Conclusion

This project compares two fundamentally different approaches to flood-risk classification.

The **Decision Tree** achieved an accuracy of approximately **99.97%** on its test set and produced a confusion matrix with **zero false negatives**.

The **KNN model** achieved approximately **90.66% accuracy**, but its confusion matrix contained **201 false negatives**, meaning a substantial number of actual flood-risk observations were classified as non-flood.

Therefore, within the specific experiments implemented in these notebooks, the Decision Tree demonstrated considerably stronger predictive results.

However, the result should be interpreted alongside the experimental limitations: the models were evaluated using different train/test splits, and the KNN implementation did not apply the imported `StandardScaler`. A controlled re-evaluation using identical data splits and preprocessing would provide a more rigorous comparison.

---

# 🛠️ Technologies & Libraries

The project uses Python and several data-science libraries:

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
* Excel dataset (`.xlsx`)

Key Scikit-learn components include:

```python
DecisionTreeClassifier
KNeighborsClassifier
train_test_split
accuracy_score
classification_report
confusion_matrix
```

---

# 📁 Project Structure

```text
.
├── Flood_prediction_using_decision_tree.ipynb
├── Flood_prediction_using_KNN_Algorithm.ipynb
├── flood_prediction_dataset.xlsx
└── README.md
```

---

# 🚀 Future Improvements

Several improvements could make this project more robust and production-oriented:

### 1. Standardize KNN features

Apply `StandardScaler` before training the KNN model.

### 2. Use the same train/test split

Both models should be evaluated using exactly the same training and testing observations.

### 3. Hyperparameter tuning

Experiment with different KNN values:

```text
K = 3
K = 5
K = 7
K = 9
...
```

For the Decision Tree, investigate parameters such as:

```text
max_depth
min_samples_split
min_samples_leaf
criterion
```

### 4. Cross-validation

Use k-fold cross-validation to obtain a more reliable estimate of model performance.

### 5. Evaluate additional metrics

Include:

* Precision
* Recall
* F1-score
* ROC-AUC
* Precision-Recall AUC

### 6. Investigate class imbalance

Since flood-risk observations represent approximately 11.39% of the dataset, techniques such as class weighting, resampling, or SMOTE could be investigated.

### 7. Feature importance

Analyze which environmental variables contribute most strongly to Decision Tree predictions.

### 8. Test additional algorithms

Future versions could compare the models against:

* Logistic Regression
* Random Forest
* Support Vector Machine
* Gradient Boosting
* XGBoost

---

# 📌 Final Takeaway

The central question of this project is:

> **Which model performs better at predicting flood risk?**

The experiments show a substantial difference between the two implemented approaches. The Decision Tree produced **99.97% test accuracy with no observed false negatives**, while KNN produced **90.66% test accuracy with 201 false negatives**.

The project therefore demonstrates why machine-learning model selection should consider **the type of errors a model makes**, not simply its overall accuracy.

---

## 👨‍💻 Project Focus

**Machine Learning | Classification | Flood Prediction | Exploratory Data Analysis | Model Evaluation | Decision Tree | K-Nearest Neighbors | Python | Scikit-learn**
