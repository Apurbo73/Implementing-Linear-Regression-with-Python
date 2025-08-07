### Implementing Linear Regression with Python
### 📘 Machine Learning: Linear Regression

**Linear Regression** is one of the simplest and most commonly used algorithms in **supervised learning**, particularly for **regression tasks**.

---

## 🔹 1. What is Linear Regression?

Linear Regression attempts to model the relationship between a **dependent variable** $y$ and one (or more) **independent variables** $x$ by fitting a **linear equation** to the observed data.

### 🔸 Simple Linear Regression Equation:

$$
y = mx + b
$$

* $y$: target (output)
* $x$: input (feature)
* $m$: slope (coefficient)
* $b$: intercept (bias)

### 🔸 Multiple Linear Regression:

$$
y = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$$

Where $w_i$ are the weights (coefficients) for each feature $x_i$.

---

## 🔹 2. Goal of Linear Regression

To find the best-fitting line (or hyperplane) by minimizing the **error** between predicted and actual values, typically using:

### 🔸 Cost Function: **Mean Squared Error (MSE)**

$$
J(w, b) = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

* $y_i$: actual value
* $\hat{y}_i$: predicted value
* $n$: number of data points

---

## 🔹 3. How It Works

1. Initialize weights and bias (e.g., to zero)
2. Make predictions: $\hat{y} = w \cdot x + b$
3. Compute the loss (e.g., MSE)
4. Use **Gradient Descent** to minimize the loss:

   * Update weights:

     $$
     w := w - \alpha \frac{\partial J}{\partial w}
     $$
   * Update bias:

     $$
     b := b - \alpha \frac{\partial J}{\partial b}
     $$
   * $\alpha$ is the learning rate

---

## 🔹 4. Code Example (Python / scikit-learn)

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 4, 2, 5, 6])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Print coefficients
print("Slope (m):", model.coef_)
print("Intercept (b):", model.intercept_)
```

---

## 🔹 5. Assumptions of Linear Regression

1. Linearity
2. Independence of errors
3. Homoscedasticity (constant variance of error terms)
4. No multicollinearity (for multiple regression)
5. Normally distributed errors

---

## 🔹 6. Use Cases

* Predicting house prices
* Forecasting sales
* Stock market trend analysis
* Estimating customer spending

---

If you’d like, I can also show:

* How to implement Linear Regression **from scratch in Python**
* How to **visualize** results
* How to handle **multiple variables** (multivariate regression)

Would you like any of these?
