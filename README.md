# Water Potability Prediction

This project aims to build a machine learning system to classify whether water is **potable (safe to drink)** based on various physicochemical features. The dataset used is real-world and contains several important indicators of water quality.

---

## Dataset: `water_potability.csv`

Each row in the dataset represents a water sample with the following attributes:

| Feature             | Description                                      |
|---------------------|--------------------------------------------------|
| `ph`               | Acidity/alkalinity of water                      |
| `Hardness`         | Concentration of calcium and magnesium           |
| `Solids`           | Total dissolved solids                           |
| `Chloramines`      | Disinfectants used for water purification        |
| `Sulfate`          | Sulfate concentration                            |
| `Conductivity`     | Ability of water to conduct electricity          |
| `Organic_carbon`   | Organic compounds in water                       |
| `Trihalomethanes`  | Potentially harmful chemical compounds           |
| `Turbidity`        | Clarity of water                                 |
| `Potability`       | Target variable: 1 if safe to drink, else 0      |

---

## Data Preprocessing

### Missing Value Handling
- Missing values in features like `ph`, `Sulfate`, `Trihalomethanes` were filled using **mean imputation**.

### Outlier Detection and Removal
- Outliers were detected using **Interquartile Range (IQR)** and removed to improve model performance.

### Feature Scaling
- Applied **Min-Max Normalization** using `MinMaxScaler`.

---

## Exploratory Data Analysis (EDA)

- **Histograms** and **boxplots** were used to explore feature distributions and detect outliers.
- **Correlation heatmaps** were generated to understand relationships between features and with the target variable.

---

## Machine Learning Models

### 1. **Neural Network (Keras)**
- Architecture: `Input → Dense(32) → Dropout → Dense(8) → Output(sigmoid)`
- Loss: `MeanSquaredError`
- Metrics: `Accuracy`
- Evaluated on accuracy, precision, recall, F1-score
- Trained with 50 epochs

### 2. **Tuned MLP Classifier (GridSearchCV)**
- Grid search over:
  - `hidden_layer_sizes`, `activation`, `solver`
- Best model was retrained and evaluated on the test set

---

### 3. **Support Vector Classifier (SVC)**
- Evaluated both default and best parameters using grid search over:
  - `C`, `gamma`, `kernel`

---

### 4. **Naive Bayes**
- Implemented using `GaussianNB`
- Evaluated on test data without hyperparameter tuning

---

### 5. **Logistic Regression**
- Grid search over:
  - `penalty`: `'l1'`, `'l2'`
  - `C`: range from `0.001` to `100`
- Evaluated using classification report and accuracy

---

### 6. **Random Forest Classifier**
- Hyperparameters tuned:
  - `n_estimators`, `max_depth`, `criterion`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`
- Best model evaluated on test set

---

### 7. **K-Nearest Neighbors (KNN)**
- Grid search over:
  - `n_neighbors`: `1 to 100`
  - `weights`: `'uniform'`, `'distance'`
  - `metric`: `'euclidean'`, `'manhattan'`

---

## Evaluation Metrics

All models were evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Printed via `classification_report`.

---

## Technologies Used

- `Python`, `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`
- `scikit-learn`, `Keras`, `TensorFlow`
- `SMOTE` for class balancing
- `GridSearchCV` for hyperparameter optimization

---



## How to Run

1. Clone the repository and install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn keras tensorflow imbalanced-learn
