# Decision Tree — Regressor & Classifier

Two standalone scripts demonstrating `DecisionTreeRegressor` and
`DecisionTreeClassifier` from scikit-learn on two different datasets.

## Files

| Script | Dataset | Model | Target |
|---|---|---|---|
| `insurance_regressor.py` | `insurance.csv` | `DecisionTreeRegressor` | `charges` (continuous) |
| `social_network_classifier.py` | `Social_Network_Ads.csv` | `DecisionTreeClassifier` | `Purchased` (0/1) |

## 1. Insurance — Decision Tree Regressor

Predicts medical insurance `charges` based on personal attributes
(age, sex, BMI, children, smoker, region).

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('insurance.csv')

lb = LabelEncoder()
df['sex'] = lb.fit_transform(df['sex'])
df['smoker'] = lb.fit_transform(df['smoker'])
df['region'] = lb.fit_transform(df['region'])

x = df.drop(columns=['charges'])
y = df['charges']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
r2_score(y_test, y_pred)
```

**Steps:**
1. Load `insurance.csv`
2. Label-encode categorical columns (`sex`, `smoker`, `region`)
3. Split features (`x`) and target (`y = charges`)
4. Train/test split (80/20)
5. Fit `DecisionTreeRegressor`
6. Evaluate with `r2_score`

## 2. Social Network Ads — Decision Tree Classifier

Predicts whether a user will purchase a product (`Purchased`: 0 or 1)
based on `Age` and `EstimatedSalary`.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('Social_Network_Ads.csv')

lb = LabelEncoder()
df['Gender'] = lb.fit_transform(df['Gender'])

x = df.drop(columns=['Purchased'])
y = df['Purchased']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
accuracy_score(y_test, y_pred)
```

**Steps:**
1. Load `Social_Network_Ads.csv`
2. Label-encode categorical columns (`Gender`)
3. Split features (`x`) and target (`y = Purchased`)
4. Train/test split (80/20)
5. Fit `DecisionTreeClassifier`
6. Evaluate with `accuracy_score` (add `precision_score`, `recall_score`,
   `f1_score`, `confusion_matrix` for a fuller picture)

## Regressor vs Classifier — what changes

| | Regressor | Classifier |
|---|---|---|
| Target type | Continuous numeric | Categorical / discrete |
| Split criterion | `squared_error` (MSE) by default | `gini` by default |
| Output | A number | A class label |
| Evaluation | `r2_score`, `mean_squared_error`, `mean_absolute_error` | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix` |

## Requirements

```
pandas
numpy
scikit-learn
```

Install with:
```
pip install pandas numpy scikit-learn
```

## Notes

- Both scripts currently use scikit-learn's default hyperparameters — no
  `max_depth`, `min_samples_split`, etc. set, so trees can overfit. Worth
  tuning (`max_depth=5`, `min_samples_leaf=10`, etc.) before trusting the
  scores on new data.
- `LabelEncoder` on multiple independent categorical columns works here,
  but for features (not the target) `OneHotEncoder` is usually the safer
  choice — `LabelEncoder` implies an ordinal relationship (0 < 1 < 2) that
  doesn't exist between e.g. regions.