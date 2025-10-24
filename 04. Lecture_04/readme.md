# Machine Learning: Comprehensive Study Notes

## Table of Contents
- [Gradient Descent](#gradient-descent)
- [Linear Regression with Multiple Variables](#linear-regression-with-multiple-variables)
- [Feature Scaling and Normalization](#feature-scaling-and-normalization)
- [Learning Rate and Debugging](#learning-rate-and-debugging)
- [Polynomial Regression](#polynomial-regression)
- [Bias-Variance Tradeoff](#bias-variance-tradeoff)
- [Logistic Regression](#logistic-regression)

---

## Gradient Descent

### Definition
**Gradient Descent** is an optimization algorithm used to minimize a cost function by iteratively moving towards the steepest descent direction. It's fundamental for training machine learning models.

### Explanation
Think of gradient descent like hiking down a mountain while blindfolded. You feel the ground around you to find the steepest downhill direction and take a small step in that direction. Repeat this process until you reach the bottom (minimum point).

The algorithm works by:
1. Starting with random parameter values
2. Calculating the gradient (slope) of the cost function
3. Updating parameters in the direction that reduces the cost
4. Repeating until convergence

### Mathematical Formulation
```
Repeat until convergence {
    θ_j := θ_j - α * ∂/∂θ_j J(θ)
}
```
Where:
- `θ_j` = parameter being optimized
- `α` = learning rate (step size)
- `J(θ)` = cost function

### Batch Gradient Descent

#### Definition
**Batch Gradient Descent** computes the gradient using the entire training dataset for each iteration.

#### Explanation
Batch GD calculates the error for each example in the training set, but only updates the model after all training examples have been evaluated. This approach provides a stable convergence path but can be computationally expensive for large datasets.

#### Characteristics
- Uses entire dataset for each update
- Stable, smooth convergence
- Computationally expensive for large datasets
- Guaranteed convergence to global minimum for convex functions

### Stochastic Gradient Descent

#### Definition
**Stochastic Gradient Descent** updates parameters for each training example one at a time.

#### Explanation
Instead of waiting to sum up the gradients over all training examples, SGD updates parameters immediately after processing each individual training example. This makes it much faster for large datasets but introduces more noise in the convergence path.

#### Characteristics
- Updates parameters after each training example
- Faster for large datasets
- Noisy, oscillating convergence path
- Can escape local minima more easily
- Requires careful learning rate scheduling

#### Comparison: Batch vs Stochastic GD

| Aspect | Batch GD | Stochastic GD |
|--------|-----------|---------------|
| Dataset Usage | Entire dataset | Single example |
| Speed | Slower | Faster |
| Convergence | Smooth | Noisy |
| Memory | Higher requirements | Lower requirements |
| Global Minima | Direct path | May oscillate |

---

## Linear Regression with Multiple Variables

### Definition
**Multiple Linear Regression** models the relationship between multiple independent variables (features) and a dependent variable (target) using a linear approach.

### Hypothesis Function
```
h_θ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```
Or in vector form:
```
h_θ(x) = θᵀx
```

### Cost Function
```
J(θ) = 1/(2m) * Σ(h_θ(xⁱ) - yⁱ)²
```
Where:
- `m` = number of training examples
- `h_θ(xⁱ)` = predicted value for i-th example
- `yⁱ` = actual value for i-th example

### Gradient Descent for Multiple Variables

#### Pseudocode
```
Initialize θ = [θ₀, θ₁, ..., θₙ] randomly
Repeat until convergence {
    for j = 0 to n:
        θ_j := θ_j - α * (1/m) * Σ(h_θ(xⁱ) - yⁱ) * x_jⁱ
}
```

#### Python Implementation
```python
import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)
        theta = theta - alpha * gradient
        cost_history[i] = compute_cost(X, y, theta)
    
    return theta, cost_history

# Example usage
X = np.array([[1, 2104, 5],    # x₀=1 (bias), x₁=size, x₂=bedrooms
              [1, 1416, 3],
              [1, 1534, 3],
              [1, 852, 2]])
y = np.array([460, 232, 315, 178])
theta = np.zeros(3)  # Initialize parameters
alpha = 0.01
iterations = 1000

theta_optimized, costs = gradient_descent(X, y, theta, alpha, iterations)
```

### Linear Regression with Normal Equations

#### Definition
The **Normal Equation** provides an analytical solution to find the optimal parameters without iterative optimization.

#### Mathematical Formulation
```
θ = (XᵀX)⁻¹Xᵀy
```

#### Python Implementation
```python
import numpy as np

def normal_equation(X, y):
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

# Example usage
X = np.array([[1, 2104, 5],
              [1, 1416, 3],
              [1, 1534, 3],
              [1, 852, 2]])
y = np.array([460, 232, 315, 178])

theta = normal_equation(X, y)
```

#### Using Scikit-Learn
```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample data
X = np.array([[2104, 5],
              [1416, 3],
              [1534, 3],
              [852, 2]])
y = np.array([460, 232, 315, 178])

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model training
model = LinearRegression()
model.fit(X_scaled, y)

# Predictions
predictions = model.predict(X_scaled)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
```

#### Comparison: Gradient Descent vs Normal Equation

| Aspect | Gradient Descent | Normal Equation |
|--------|------------------|-----------------|
| Learning Rate | Required | Not needed |
| Iterations | Many iterations needed | No iterations |
| Complexity | O(kn²) | O(n³) |
| Large n | Works well | Slow for n > 10,000 |
| Feature Scaling | Needed | Not needed |

---

## Feature Scaling and Normalization

### Feature Scaling

#### Definition
**Feature Scaling** is the process of normalizing the range of independent variables or features of data.

#### Explanation
When features have different scales, the gradient descent algorithm may take longer to converge. Features with larger ranges can dominate the cost function and slow down learning.

#### Methods
1. **Min-Max Scaling**: `x' = (x - min(x)) / (max(x) - min(x))`
2. **Standardization**: `x' = (x - μ) / σ`

#### Example
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Sample data
X = np.array([[2104, 5],
              [1416, 3],
              [1534, 3],
              [852, 2]])

# Standardization
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)

# Min-Max Scaling
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)
```

### Mean Normalization

#### Definition
**Mean Normalization** adjusts features to have approximately zero mean.

#### Formula
```
x' = (x - μ) / (max(x) - min(x))
```
Or for a specific range:
```
x₁ = (size - 1000) / 2000
x₂ = (#bedrooms - 2) / 5
```

#### Python Implementation
```python
def mean_normalization(X):
    mean = np.mean(X, axis=0)
    range_vals = np.max(X, axis=0) - np.min(X, axis=0)
    X_normalized = (X - mean) / range_vals
    return X_normalized

# Example
X = np.array([[2104, 5],
              [1416, 3],
              [1534, 3],
              [852, 2]])
X_normalized = mean_normalization(X)
```

---

## Learning Rate and Debugging

### Learning Rate

#### Definition
The **Learning Rate (α)** determines the size of steps taken during gradient descent optimization.

#### Explanation
Think of learning rate as the step size when walking down a hill:
- Too small: Very slow convergence
- Too large: May overshoot the minimum and fail to converge
- Just right: Efficient convergence to minimum

### Debugging Gradient Descent

#### Convergence Monitoring

```python
def gradient_descent_with_debugging(X, y, theta, alpha, iterations, convergence_threshold=1e-3):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors)
        theta = theta - alpha * gradient
        
        current_cost = compute_cost(X, y, theta)
        cost_history.append(current_cost)
        
        # Check for convergence
        if i > 0 and abs(cost_history[i-1] - current_cost) < convergence_threshold:
            print(f"Converged after {i} iterations")
            break
    
    return theta, cost_history
```

#### Learning Rate Selection

**Guidelines for choosing α:**
- Try values: ..., 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, ...
- Plot cost function vs iterations for different α values
- Choose α where cost decreases steadily

#### Symptoms of Poor Learning Rate

**α too small:**
- Very slow decrease in cost function
- Requires many iterations to converge

**α too large:**
- Cost function may increase or oscillate
- May never converge

### Automatic Convergence Test
Declare convergence when `J(θ)` decreases by less than `10⁻³` in one iteration.

---

## Polynomial Regression

### Definition
**Polynomial Regression** fits a nonlinear relationship between independent and dependent variables by adding powers of the original features.

### Explanation
When data shows curved patterns, linear regression may underfit. Polynomial regression addresses this by creating new features that are powers of existing features.

### Mathematical Formulation
**Linear Equation**: `y = θ₀ + θ₁x`  
**Quadratic Equation**: `y = θ₀ + θ₁x + θ₂x²`  
**Cubic Equation**: `y = θ₀ + θ₁x + θ₂x² + θ₃x³`

### Python Implementation
```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.linspace(-3, 3, 100)
y = X**3 - 2*X**2 + X + np.random.normal(0, 2, 100)

# Reshape for sklearn
X = X.reshape(-1, 1)

# Polynomial features
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Linear regression on polynomial features
model = LinearRegression()
model.fit(X_poly, y)

# Predictions
y_pred = model.predict(X_poly)

# Evaluation
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"RMSE: {np.sqrt(mse):.2f}")
print(f"R² Score: {r2:.2f}")
```

### Feature Engineering for Polynomial Regression
```python
# Manual feature engineering for housing example
def create_polynomial_features(X):
    X_poly = np.column_stack([
        X[:, 0],                    # size
        X[:, 0]**2,                 # size²
        X[:, 0]**3,                 # size³
        X[:, 1],                    # bedrooms
        X[:, 0] * X[:, 1]           # interaction term
    ])
    return X_poly

# Example usage
X_original = np.array([[2104, 5],
                       [1416, 3],
                       [1534, 3],
                       [852, 2]])
X_polynomial = create_polynomial_features(X_original)
```

---

## Bias-Variance Tradeoff

### Underfitting and Overfitting

#### Underfitting
**Definition**: When a model is too simple to capture the underlying pattern in the data.

**Characteristics**:
- High bias, low variance
- Poor performance on both training and test data
- Model is too simplistic

**Example**: Using linear regression for clearly nonlinear data.

#### Overfitting
**Definition**: When a model is too complex and learns the noise in the training data.

**Characteristics**:
- Low bias, high variance
- Excellent performance on training data, poor on test data
- Model is too complex

**Example**: Using a 20th-degree polynomial that passes through every training point.

### Bias vs Variance

#### Bias
**Definition**: Error due to simplistic assumptions in the model.

**High Bias**:
- Underfitting
- Unable to capture data patterns
- Consistent inaccuracies

#### Variance
**Definition**: Error due to complex model trying to fit the data too closely.

**High Variance**:
- Overfitting
- Sensitive to small fluctuations in training data
- Inconsistent predictions

### The Tradeoff
```python
import matplotlib.pyplot as plt
import numpy as np

# Simulate bias-variance tradeoff
degrees = [1, 2, 3, 10, 20]
train_errors = []
test_errors = []

for degree in degrees:
    # Model training and evaluation code here
    # Typically, as degree increases:
    # - Training error decreases (lower bias)
    # - Test error decreases then increases (variance increases)
    pass

# Plotting the tradeoff
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, label='Training Error', marker='o')
plt.plot(degrees, test_errors, label='Test Error', marker='s')
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('Error')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.grid(True)
plt.show()
```

### Managing Bias-Variance Tradeoff

**Strategies for High Bias (Underfitting)**:
- Add more features
- Use more complex model
- Reduce regularization

**Strategies for High Variance (Overfitting)**:
- Get more training data
- Use simpler model
- Increase regularization
- Feature selection
- Cross-validation

---

## Logistic Regression

### Classification

#### Definition
**Classification** is a supervised learning task where the output variable is categorical.

#### Examples
- Email: Spam / Not Spam
- Transactions: Fraudulent / Not Fraudulent  
- Tumor: Malignant / Benign

#### Binary Classification
```
y ∈ {0, 1}
```
Where:
- 0: "Negative Class" (e.g., benign tumor)
- 1: "Positive Class" (e.g., malignant tumor)

### Binary Logistic Regression

#### Definition
**Logistic Regression** is a classification algorithm used when the dependent variable is categorical. For binary outcomes, it's called **Binary Logistic Regression**.

#### Why Not Linear Regression?
Linear regression outputs can be >1 or <0, but we need probabilities between 0 and 1 for classification.

### Activation Functions

#### Definition
**Activation Functions** transform the weighted sum of inputs to produce an output within a specific range.

#### Linear Activation Function
```
g(z) = z
```
- No change to output
- Used in linear regression

#### Rectified Linear Activation (ReLU)
```
g(z) = max(0, z)
```
- Outputs 0 for negative inputs, linear for positive
- Commonly used in deep learning

#### Sigmoid Activation Function
```
g(z) = 1 / (1 + e^(-z))
```
- Outputs between 0 and 1
- Perfect for binary classification
- Also called logistic function

#### Hyperbolic Tangent (tanh)
```
g(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```
- Outputs between -1 and 1
- Zero-centered

### Python Implementation of Activation Functions
```python
import numpy as np
import matplotlib.pyplot as plt

def linear(z):
    return z

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

# Plotting
z = np.linspace(-5, 5, 100)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(z, linear(z))
plt.title('Linear Activation')

plt.subplot(2, 2, 2)
plt.plot(z, relu(z))
plt.title('ReLU Activation')

plt.subplot(2, 2, 3)
plt.plot(z, sigmoid(z))
plt.title('Sigmoid Activation')

plt.subplot(2, 2, 4)
plt.plot(z, tanh(z))
plt.title('Tanh Activation')

plt.tight_layout()
plt.show()
```

### Hypothesis Representation

#### Logistic Regression Model
```
h_θ(x) = g(θᵀx) = 1 / (1 + e^(-θᵀx))
```

#### Interpretation
`h_θ(x)` represents the estimated probability that `y = 1` given input `x`:
```
h_θ(x) = P(y = 1 | x; θ)
```
Example: If `h_θ(x) = 0.7`, there's a 70% chance the tumor is malignant.

#### Python Implementation
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_hypothesis(X, theta):
    return sigmoid(X.dot(theta))

# Example usage
X = np.array([[1, 0.5, 0.3],   # x₀=1, x₁, x₂
              [1, 0.8, 0.2],
              [1, 0.2, 0.9]])
theta = np.array([-1, 2, 1])    # Parameters

probabilities = logistic_hypothesis(X, theta)
print("Probabilities:", probabilities)
```

### Decision Boundary

#### Definition
The **Decision Boundary** is the line/plane that separates different classes predicted by the model.

#### Linear Decision Boundary
For `h_θ(x) = g(θ₀ + θ₁x₁ + θ₂x₂)`, the decision boundary is:
```
θ₀ + θ₁x₁ + θ₂x₂ = 0
```
- Predict `y = 1` if `θᵀx ≥ 0`
- Predict `y = 0` if `θᵀx < 0`

#### Example
If `θ = [-3, 1, 1]`, decision boundary is:
```
-3 + x₁ + x₂ = 0  =>  x₁ + x₂ = 3
```

#### Non-linear Decision Boundary
```python
# Example: Circular decision boundary
def non_linear_boundary(X):
    # Add polynomial features
    X_poly = np.column_stack([
        X[:, 0],           # x₁
        X[:, 1],           # x₂  
        X[:, 0]**2,        # x₁²
        X[:, 1]**2         # x₂²
    ])
    return X_poly

# With θ = [-1, 0, 0, 1, 1], decision boundary is:
# -1 + x₁² + x₂² = 0  =>  x₁² + x₂² = 1 (circle)
```

### Cost Function for Logistic Regression

#### Problem with MSE
Mean Squared Error doesn't work well with logistic regression because the cost function becomes non-convex due to the sigmoid function.

#### Logistic Regression Cost Function
```
Cost(h_θ(x), y) = -log(h_θ(x))     if y = 1
                 -log(1 - h_θ(x))   if y = 0
```

#### Combined Form
```
Cost(h_θ(x), y) = -y log(h_θ(x)) - (1 - y) log(1 - h_θ(x))
```

#### Complete Cost Function
```
J(θ) = -1/m * Σ [yⁱ log(h_θ(xⁱ)) + (1 - yⁱ) log(1 - h_θ(xⁱ))]
```

#### Python Implementation
```python
def logistic_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    
    # Avoid log(0) errors
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    
    cost = - (1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost
```

### Gradient Descent for Logistic Regression

#### Pseudocode
```
Initialize θ
Repeat until convergence {
    θ_j := θ_j - α * (1/m) * Σ (h_θ(xⁱ) - yⁱ) * x_jⁱ
    (simultaneously update all θ_j)
}
```

#### Python Implementation
```python
def logistic_gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        h = sigmoid(X.dot(theta))
        gradient = (1/m) * X.T.dot(h - y)
        theta = theta - alpha * gradient
        
        cost = logistic_cost(X, y, theta)
        cost_history.append(cost)
        
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost:.4f}")
    
    return theta, cost_history

# Example usage
X = np.array([[1, 0.5, 0.3],
              [1, 0.8, 0.2], 
              [1, 0.2, 0.9],
              [1, 0.9, 0.1]])
y = np.array([1, 1, 0, 1])
theta = np.zeros(3)
alpha = 0.1
iterations = 1000

theta_optimized, costs = logistic_gradient_descent(X, y, theta, alpha, iterations)
```

### Complete Logistic Regression with Scikit-Learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Sample data: exam scores and admission results
X = np.array([[85, 90], [60, 75], [45, 50], [95, 85], 
              [70, 65], [55, 45], [80, 80], [65, 70]])
y = np.array([1, 1, 0, 1, 0, 0, 1, 0])  # 1=admitted, 0=not admitted

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Decision boundary visualization
def plot_decision_boundary(X, y, model, scaler):
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict for each point in mesh grid
    Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title('Logistic Regression Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot decision boundary
plot_decision_boundary(X, y, model, scaler)
```
