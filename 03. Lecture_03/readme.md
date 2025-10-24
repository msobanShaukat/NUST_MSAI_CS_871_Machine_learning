## SUPERVISED LEARNING

### Data Representation

**Definition:** The process of converting raw data into formats that machine learning algorithms can effectively process and learn from.

**Explanation:** Data representation is crucial because most ML algorithms require numerical input. This involves transforming various data types (text, categories, images) into numerical formats while preserving meaningful information. Good representation makes patterns easier for algorithms to detect.

**Examples:**
- Converting text to word frequency vectors
- Transforming categories to numerical codes
- Normalizing numerical values to common scales
- Creating feature combinations

**Representation Techniques:**
- One-hot encoding for categorical data
- TF-IDF for text data
- Normalization and standardization for numerical data
- Feature scaling and transformation

---

### Handling Non-Numeric Data

**Definition:** Techniques for converting categorical, textual, or other non-numerical data into numerical formats suitable for machine learning algorithms.

**Explanation:** Most ML algorithms work with numbers, so we need methods to convert non-numeric data while preserving meaningful relationships. Different types of non-numeric data require different encoding strategies to ensure the model can learn effectively from them.

**Examples:**
- **Categorical Data:** Colors, countries, types
- **Text Data:** Reviews, descriptions, documents
- **Date/Time Data:** Timestamps, durations
- **Ordinal Data:** Ratings, levels (low/medium/high)

**Encoding Methods:**
- **Label Encoding:** Assigning numbers to categories (Red=1, Blue=2, Green=3)
- **One-Hot Encoding:** Creating binary columns for each category
- **Ordinal Encoding:** Preserving order in ranked categories
- **Target Encoding:** Using target statistics for categories

---

### Data Representation and Linear Algebra

**Definition:** Using linear algebra concepts to represent and manipulate data in machine learning, particularly through vectors and matrices.

**Explanation:** Linear algebra provides the mathematical foundation for data representation in ML. Data points are represented as vectors, datasets as matrices, and operations like transformations and similarities are computed using linear algebra. This mathematical framework enables efficient computation and conceptual understanding.

**Examples:**
- Representing a house as a vector: [price, bedrooms, sq_ft, location]
- Storing a dataset as a matrix where rows are samples and columns are features
- Using dot products to compute similarities
- Applying matrix transformations for feature engineering

**Key Concepts:**
- **Vectors:** Represent individual data points
- **Matrices:** Represent entire datasets
- **Dot Products:** Measure similarity between vectors
- **Matrix Operations:** Enable efficient batch processing

---

## LINEAR REGRESSION (SINGLE VARIABLE)

### Hypothesis

**Definition:** In linear regression, the hypothesis is the proposed linear relationship between the input variable and output variable that the model tries to learn.

**Explanation:** The hypothesis represents our model's prediction function. For single variable linear regression, it's a straight line equation: y = mx + b, where:
- y is the predicted output
- x is the input feature
- m is the slope (weight)
- b is the y-intercept (bias)

The model's goal is to find the best values for m and b that minimize prediction errors.

**Mathematical Form:**
```
hθ(x) = θ₀ + θ₁x
```
Where:
- hθ(x) is the hypothesis function
- θ₀ is the bias term (y-intercept)
- θ₁ is the weight for feature x

**Example:**
For house price prediction:
- x = house size (sq ft)
- hθ(x) = predicted price
- θ₀ = base price (when size is 0)
- θ₁ = price per square foot

---

### Parameters

**Definition:** The values in the hypothesis function that the learning algorithm adjusts during training to fit the data.

**Explanation:** Parameters (often called weights) are the variables that define our model. In single variable linear regression, we have two parameters: the slope (θ₁) and intercept (θ₀). During training, the algorithm systematically adjusts these parameters to find the line that best fits the training data.

**Key Parameters:**
- **θ₀ (Intercept/Bias):** The predicted value when all input features are zero
- **θ₁ (Slope/Weight):** How much the output changes for a one-unit change in the input

**Learning Process:**
1. Start with random parameter values
2. Calculate predictions using current parameters
3. Measure how wrong predictions are (cost)
4. Update parameters to reduce cost
5. Repeat until optimal parameters are found

---

### Cost Function

**Definition:** A function that measures how wrong the model's predictions are, used to guide the learning algorithm toward better parameters.

**Explanation:** The cost function quantifies the error between predicted values and actual values. For linear regression, we typically use Mean Squared Error (MSE), which calculates the average squared difference between predictions and actual values. The learning algorithm's goal is to find parameters that minimize this cost function.

**Mathematical Form (MSE):**
```
J(θ₀, θ₁) = (1/2m) * Σ(hθ(xⁱ) - yⁱ)²
```
Where:
- J is the cost function
- m is the number of training examples
- hθ(xⁱ) is the prediction for example i
- yⁱ is the actual value for example i

**Example:**
If our model predicts house prices and:
- Actual prices: [200k, 300k, 400k]
- Predicted prices: [210k, 290k, 390k]
- Cost = average of [(210-200)², (290-300)², (390-400)²] = average of [100, 100, 100] = 100

---

### Goal

**Definition:** The objective of linear regression is to find the parameter values that minimize the cost function, resulting in the best-fitting line through the data.

**Explanation:** The ultimate goal is to learn the relationship between input and output variables so we can make accurate predictions on new, unseen data. This involves finding the optimal balance where the line is close to all data points without being too sensitive to individual outliers.

**Primary Objectives:**
1. **Minimize Prediction Error:** Find parameters that make predictions as close as possible to actual values
2. **Generalize Well:** Ensure the model works on new data, not just training data
3. **Understand Relationships:** Quantify how input changes affect output

**Success Metrics:**
- Low cost function value on training data
- Good performance on test data (generalization)
- Meaningful parameter interpretations

---

## GRADIENT DESCENT

**Definition:** An optimization algorithm used to minimize the cost function by iteratively adjusting parameters in the direction of steepest descent.

**Explanation:** Gradient descent is like finding the bottom of a valley by always walking downhill. The algorithm calculates the gradient (slope) of the cost function with respect to each parameter, then updates the parameters by moving a small step in the direction that reduces the cost most. This process repeats until reaching the minimum cost.

**Mathematical Update Rule:**
```
θⱼ := θⱼ - α * ∂/∂θⱼ J(θ₀, θ₁)
```
Where:
- θⱼ is the parameter being updated
- α is the learning rate (step size)
- ∂/∂θⱼ J(θ₀, θ₁) is the gradient (slope) of cost function

**Step-by-Step Process:**
1. Initialize parameters with random values
2. Calculate cost and gradients
3. Update parameters: move opposite to gradient direction
4. Repeat until convergence (minimal cost improvement)

**Key Concepts:**
- **Learning Rate (α):** Controls step size - too small (slow convergence), too large (may overshoot)
- **Gradient:** Direction and steepness of cost increase
- **Convergence:** Reaching the minimum cost point

**Example:**
Finding the best line for house price prediction:
- Start with random slope and intercept
- Calculate how wrong predictions are
- Adjust slope and intercept to reduce error
- Repeat until predictions are as accurate as possible

---

## LINEAR REGRESSION IMPLEMENTATION

### Linear Regression (Single Variable) Pseudo Code

```python
# PSEUDO CODE FOR SINGLE VARIABLE LINEAR REGRESSION

Initialize parameters:
    theta0 = random_value
    theta1 = random_value
    learning_rate = 0.01
    iterations = 1000

Define hypothesis function:
    h(x) = theta0 + theta1 * x

Define cost function:
    cost = 0
    for i in range(m):  # m = number of training examples
        cost += (h(x[i]) - y[i]) ** 2
    cost = cost / (2 * m)

Gradient Descent Algorithm:
for iteration in range(iterations):
    # Initialize gradients
    grad0 = 0
    grad1 = 0
    
    # Calculate gradients
    for i in range(m):
        error = h(x[i]) - y[i]
        grad0 += error
        grad1 += error * x[i]
    
    # Average gradients
    grad0 = grad0 / m
    grad1 = grad1 / m
    
    # Update parameters
    theta0 = theta0 - learning_rate * grad0
    theta1 = theta1 - learning_rate * grad1
    
    # Optional: Print cost every 100 iterations
    if iteration % 100 == 0:
        current_cost = calculate_cost(x, y, theta0, theta1)
        print(f"Iteration {iteration}, Cost: {current_cost}")

Return final parameters:
    return theta0, theta1
```

### Linear Regression (Single Variable) in Python

```python
# LINEAR REGRESSION FROM SCRATCH IN PYTHON

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta0 = None
        self.theta1 = None
        self.cost_history = []
    
    def hypothesis(self, x):
        """Calculate prediction: h(x) = theta0 + theta1 * x"""
        return self.theta0 + self.theta1 * x
    
    def compute_cost(self, x, y):
        """Calculate Mean Squared Error cost"""
        m = len(y)
        predictions = self.hypothesis(x)
        cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
        return cost
    
    def fit(self, x, y):
        """Train the model using gradient descent"""
        m = len(y)
        
        # Initialize parameters
        self.theta0 = 0
        self.theta1 = 0
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.iterations):
            # Calculate predictions
            predictions = self.hypothesis(x)
            
            # Calculate gradients
            grad0 = (1/m) * np.sum(predictions - y)
            grad1 = (1/m) * np.sum((predictions - y) * x)
            
            # Update parameters
            self.theta0 = self.theta0 - self.learning_rate * grad0
            self.theta1 = self.theta1 - self.learning_rate * grad1
            
            # Record cost for plotting
            cost = self.compute_cost(x, y)
            self.cost_history.append(cost)
            
            # Print progress
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")
    
    def predict(self, x):
        """Make predictions using learned parameters"""
        return self.hypothesis(x)

# Example usage
if __name__ == "__main__":
    # Sample data: house sizes (sq ft) vs prices
    house_sizes = np.array([1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700])
    house_prices = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])
    
    # Create and train model
    model = LinearRegression(learning_rate=0.0000001, iterations=1000)
    model.fit(house_sizes, house_prices)
    
    # Make prediction
    new_house_size = 2000
    predicted_price = model.predict(new_house_size)
    print(f"Predicted price for {new_house_size} sq ft house: ${predicted_price:,.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Regression line
    plt.subplot(1, 2, 1)
    plt.scatter(house_sizes, house_prices, alpha=0.7)
    plt.plot(house_sizes, model.predict(house_sizes), color='red')
    plt.xlabel('House Size (sq ft)')
    plt.ylabel('Price ($)')
    plt.title('Linear Regression: House Size vs Price')
    plt.grid(True)
    
    # Plot 2: Cost history
    plt.subplot(1, 2, 2)
    plt.plot(model.cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function Over Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
```

### Linear Regression with Built-in Libraries (Scikit-learn)

```python
# LINEAR REGRESSION USING SCIKIT-LEARN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Sample data preparation
# House sizes (sq ft) and corresponding prices
X = np.array([1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]).reshape(-1, 1)
y = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model evaluation
print("=== MODEL EVALUATION ===")
print(f"Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"R-squared score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

# Make prediction for new data
new_house_size = np.array([[2000]])
predicted_price = model.predict(new_house_size)
print(f"\nPredicted price for 2000 sq ft house: ${predicted_price[0]:,.2f}")

# Visualization
plt.figure(figsize=(10, 6))

# Plot training data
plt.scatter(X_train, y_train, color='blue', alpha=0.7, label='Training Data')

# Plot test data
plt.scatter(X_test, y_test, color='green', alpha=0.7, label='Test Data')

# Plot regression line
x_line = np.linspace(1000, 2500, 100).reshape(-1, 1)
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color='red', linewidth=2, label='Regression Line')

plt.xlabel('House Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Linear Regression with Scikit-learn: House Size vs Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Additional: Model interpretation
print("\n=== MODEL INTERPRETATION ===")
print(f"The model suggests that each additional square foot adds ${model.coef_[0]:.2f} to the house price")
print(f"The base price (when size is 0) is estimated at ${model.intercept_:.2f}")

# Cross-validation example
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"\n=== CROSS-VALIDATION RESULTS ===")
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Average R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```
