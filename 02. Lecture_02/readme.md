# Machine Learning - MSc Level Comprehensive Notes

## Table of Contents
1. [Machine Learning Methods](#machine-learning-methods)
2. [Machine Learning Design Cycle](#machine-learning-design-cycle)
3. [AI vs ML vs DL](#ai-vs-ml-vs-dl)
4. [Types of Machine Learning](#types-of-machine-learning)
5. [Supervised Learning](#supervised-learning)
6. [Linear Regression](#linear-regression)

---

## MACHINE LEARNING METHODS

### Template Matching and Issues

**Definition:** Template matching is a machine learning approach where new data is classified by comparing it against stored templates or prototypes of each class.

**Explanation:** Think of template matching like using stencils. You have pre-made templates for different categories (like letters, shapes, or objects), and when new data comes in, you check which template it matches most closely. It's essentially pattern recognition where you measure the similarity between the input and your stored patterns.

**Examples:**
- Character recognition: Comparing a handwritten digit to templates of digits 0-9
- Face detection: Matching facial features against template face patterns
- Object recognition: Identifying objects by comparing with template shapes

**Issues and Concerns:**
- **Scalability Problems:** As the number of classes increases, you need more templates, making it computationally expensive
- **Variability Handling:** Poor performance when the same object appears in different orientations, sizes, or lighting conditions
- **Memory Intensive:** Requires storing multiple templates for each class to handle variations
- **Rigid Structure:** Cannot handle deformations or partial occlusions well

---

### Statistical Approach and Issues

**Definition:** The statistical approach uses probability theory and statistical models to make predictions and decisions based on data patterns.

**Explanation:** This method treats machine learning as a statistical inference problem. Instead of looking for exact matches, it calculates probabilities. For example, "Given these features, what's the probability this email is spam?" It uses mathematical models to find relationships between variables and make predictions based on statistical patterns.

**Examples:**
- Spam filtering: Calculating probability an email is spam based on word frequencies
- Medical diagnosis: Predicting disease likelihood based on symptoms and test results
- Weather forecasting: Predicting rain probability based on atmospheric conditions

**Issues and Concerns:**
- **Distribution Assumptions:** Often assumes data follows specific distributions (like normal distribution) which may not be true
- **Over-reliance on Historical Data:** If the underlying data distribution changes, models can become inaccurate
- **Feature Independence Assumptions:** Many models assume features are independent, which is rarely true in real world
- **Curse of Dimensionality:** Performance degrades as the number of features increases

---

### Syntactic Approach and Issues

**Definition:** The syntactic approach uses formal grammar and structural relationships to recognize and classify patterns based on their composition and arrangement.

**Explanation:** This method treats patterns as sentences in a language, where primitive elements combine according to grammatical rules. Instead of looking at overall similarity, it analyzes how components are structured and related. It's like understanding a sentence by analyzing its grammatical structure rather than just matching words.

**Examples:**
- Chemical compound analysis: Recognizing molecular structures based on atomic arrangements
- Scene understanding: Analyzing how objects in an image relate to each other spatially
- Speech recognition: Understanding language structure beyond just word matching

**Issues and Concerns:**
- **Complex Rule Creation:** Requires extensive domain knowledge to create accurate grammatical rules
- **Computational Complexity:** Parsing and analyzing structures can be computationally intensive
- **Limited Flexibility:** Struggles with patterns that don't strictly follow the predefined grammar
- **Difficulty in Learning:** Hard to automatically learn syntactic rules from data

---

### Neural Networks and Issues

**Definition:** Neural networks are computing systems inspired by biological brains that learn to recognize patterns through training on examples.

**Explanation:** Neural networks consist of interconnected nodes (neurons) organized in layers. Each connection has a weight that adjusts during learning. The network processes input data through these layers, with each neuron performing simple computations, ultimately producing an output. Through training, the network learns which patterns in the data are important for making accurate predictions.

**Examples:**
- Image classification: Identifying objects in photos
- Speech recognition: Converting spoken words to text
- Game playing: Learning strategies through practice

**Issues and Concerns:**
- **Black Box Nature:** Difficult to interpret how decisions are made
- **Data Hungry:** Require large amounts of training data
- **Computational Resources:** Training can be computationally expensive
- **Overfitting Risk:** Can memorize training data rather than learning general patterns
- **Hyperparameter Sensitivity:** Performance highly dependent on proper configuration

---

## MACHINE LEARNING DESIGN CYCLE

### Data Collection

**Definition:** The process of gathering and measuring information on variables of interest to build datasets for training machine learning models.

**Explanation:** Data collection is the foundation of any machine learning project. You need to collect relevant, high-quality data that represents the problem you're trying to solve. This involves identifying what data you need, where to get it, and how to gather it systematically. The quality of your data directly impacts your model's performance.

**Examples:**
- Collecting customer purchase history for recommendation systems
- Gathering medical images for disease detection models
- Compiling sensor readings for predictive maintenance

**Best Practices:**
- Ensure data represents real-world scenarios
- Collect sufficient quantity for reliable learning
- Maintain data quality and consistency
- Consider privacy and ethical implications

---

### Feature Choice

**Definition:** The process of selecting, creating, and transforming variables that will be used as inputs for machine learning algorithms.

**Explanation:** Features are the characteristics or properties of your data that the model uses to make predictions. Good feature choice means selecting the most relevant, informative, and non-redundant variables. This can involve feature selection (choosing existing features) and feature engineering (creating new features from existing ones).

**Examples:**
- For spam detection: choosing word frequencies, sender information, email structure
- For house price prediction: selecting square footage, location, number of bedrooms
- For image recognition: using pixel values, edges, textures as features

**Key Considerations:**
- Relevance to prediction task
- Independence from other features
- Computational efficiency
- Interpretability

---

### Model Choice

**Definition:** Selecting the appropriate machine learning algorithm that best fits the problem, data characteristics, and project requirements.

**Explanation:** Different machine learning algorithms have different strengths, weaknesses, and assumptions. The choice depends on factors like the type of problem (classification, regression, clustering), data size and quality, required accuracy, interpretability needs, and computational constraints.

**Examples:**
- **Linear Regression:** For predicting continuous values with linear relationships
- **Decision Trees:** For interpretable classification with clear rules
- **Neural Networks:** For complex pattern recognition in large datasets
- **K-Means:** For grouping similar data points without labels

**Selection Criteria:**
- Problem type (supervised vs unsupervised)
- Data size and dimensionality
- Training time constraints
- Interpretability requirements
- Expected performance

---

### Training

**Definition:** The process where a machine learning model learns patterns and relationships from training data by adjusting its internal parameters.

**Explanation:** During training, the model is exposed to labeled examples (in supervised learning) or patterns (in unsupervised learning). The algorithm iteratively adjusts its parameters to minimize the difference between its predictions and the actual outcomes. This process continues until the model achieves satisfactory performance or meets stopping criteria.

**Examples:**
- Adjusting weights in a neural network to recognize cat images
- Finding the best split points in a decision tree
- Learning cluster centers in K-means clustering

**Training Process Steps:**
1. Initialize model parameters
2. Make predictions on training data
3. Calculate error between predictions and actual values
4. Update parameters to reduce error
5. Repeat until convergence

---

### Evaluation

**Definition:** The process of assessing a trained model's performance using metrics and test data to ensure it generalizes well to new, unseen data.

**Explanation:** Evaluation tells you how well your model will perform in the real world. It involves testing the model on data it hasn't seen during training and using appropriate metrics to measure its performance. This helps identify issues like overfitting and ensures the model is ready for deployment.

**Examples:**
- Measuring accuracy on a test set of images
- Calculating precision and recall for spam detection
- Computing mean squared error for price predictions

**Common Evaluation Metrics:**
- **Classification:** Accuracy, Precision, Recall, F1-score
- **Regression:** Mean Absolute Error, Mean Squared Error, R-squared
- **Clustering:** Silhouette Score, Davies-Bouldin Index

---

### Computational Complexity

**Definition:** The analysis of the resources (time and memory) required by machine learning algorithms as the size of input data increases.

**Explanation:** Computational complexity helps us understand how an algorithm's performance scales with larger datasets. It's crucial for choosing appropriate algorithms for large-scale problems and estimating resource requirements. Complexity is typically expressed using Big O notation.

**Examples:**
- Linear regression: O(n²p + p³) where n is samples, p is features
- K-means clustering: O(nkdi) where n is points, k is clusters, d is dimensions, i is iterations
- Neural networks: Varies greatly with architecture and data

**Complexity Types:**
- **Time Complexity:** How runtime increases with input size
- **Space Complexity:** How memory usage increases with input size
- **Sample Complexity:** How much data is needed to learn effectively

---

## AI VS ML VS DL

### Definitions and Relationships

**Artificial Intelligence (AI):**
**Definition:** The broad science of creating intelligent machines that can perform tasks typically requiring human intelligence.

**Explanation:** AI is the overarching field that encompasses any technique enabling computers to mimic human intelligence. This includes reasoning, learning, problem-solving, perception, and language understanding. AI can be rule-based systems, expert systems, or learning systems.

**Examples:**
- Chess-playing computers
- Voice assistants like Siri and Alexa
- Self-driving cars
- Recommendation systems

**Machine Learning (ML):**
**Definition:** A subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.

**Explanation:** ML focuses on developing algorithms that can learn patterns from data and make decisions or predictions. Instead of following predetermined rules, ML models learn these rules from examples. It's the primary approach used in modern AI systems.

**Examples:**
- Spam filters learning from labeled emails
- Fraud detection systems learning from transaction patterns
- Movie recommendation engines learning from viewing history

**Deep Learning (DL):**
**Definition:** A subset of machine learning that uses artificial neural networks with multiple layers to learn representations of data.

**Explanation:** Deep learning uses neural networks with many layers (hence "deep") to automatically learn hierarchical representations of data. Lower layers might learn simple features like edges, while higher layers learn more complex concepts. DL has revolutionized fields like computer vision and natural language processing.

**Examples:**
- Image recognition in photos
- Real-time speech translation
- Generating realistic images and text

**Relationship Hierarchy:**
```
Artificial Intelligence (AI)
    ↳ Machine Learning (ML)
        ↳ Deep Learning (DL)
```

**Key Differences:**
- **AI:** Any technique for intelligent behavior
- **ML:** Learning from data without explicit programming
- **DL:** Learning through deep neural networks with multiple layers

---

## TYPES OF MACHINE LEARNING

### Supervised Learning

**Definition:** A type of machine learning where the model is trained on labeled data, meaning each training example is paired with the correct output.

**Explanation:** In supervised learning, you provide the algorithm with input-output pairs during training. The model learns to map inputs to outputs by finding patterns in this labeled data. Once trained, it can predict outputs for new, unseen inputs. It's called "supervised" because the training process is guided by knowing the right answers.

**Examples:**
- Email spam classification (input: email, output: spam/not spam)
- House price prediction (input: house features, output: price)
- Medical diagnosis (input: symptoms, output: disease)

**Common Algorithms:**
- Linear Regression
- Logistic Regression
- Decision Trees
- Support Vector Machines
- Neural Networks

---

### Unsupervised Learning

**Definition:** A type of machine learning where the model works with unlabeled data to find hidden patterns or intrinsic structures in the input data.

**Explanation:** Unsupervised learning deals with data that has no labels or predefined categories. The algorithm must discover the underlying structure on its own by identifying similarities, patterns, or groupings in the data. It's like exploring a new city without a map and discovering neighborhoods based on building styles.

**Examples:**
- Customer segmentation based on purchasing behavior
- Grouping similar documents together
- Anomaly detection in network security
- Reducing data dimensionality for visualization

**Common Algorithms:**
- K-means Clustering
- Hierarchical Clustering
- Principal Component Analysis (PCA)
- Association Rules

---

### Reinforcement Learning

**Definition:** A type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward.

**Explanation:** In reinforcement learning, an agent interacts with an environment, takes actions, and receives rewards or penalties. The agent learns through trial and error which actions yield the best long-term rewards. Unlike supervised learning, there's no labeled dataset - the learning comes from the consequences of actions.

**Examples:**
- Teaching a robot to walk
- Game playing AI (like AlphaGo)
- Autonomous vehicle navigation
- Resource management in data centers

**Key Components:**
- **Agent:** The learner or decision maker
- **Environment:** Everything the agent interacts with
- **Actions:** What the agent can do
- **Rewards:** Feedback from the environment
- **Policy:** Strategy that the agent follows

---

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

## Summary

These comprehensive notes cover the fundamental concepts of machine learning at an MSc level, focusing on:

1. **Different ML approaches** and their trade-offs
2. **The complete ML design cycle** from data collection to evaluation
3. **Clear distinctions** between AI, ML, and DL
4. **All major learning paradigms** (supervised, unsupervised, reinforcement)
5. **Practical implementation** of linear regression from scratch and using libraries

The notes provide both theoretical understanding and practical implementation skills, with detailed explanations, examples, and code implementations to ensure comprehensive learning.
