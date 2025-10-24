---

# **CS-817 Machine Learning: Logistic Regression with Regularization**
## **MSc in Artificial Intelligence and Data Sciences**

<div align="center">

https://www.perplexity.ai/search/master-prompt-creating-compreh-KgeaRraiQsWdcDKCIejtyQ#2

![Machine Learning](https://img.shields.


![Level](https://img.shields.io/badge/Level-Masters-green?style=for-the-badge/badge/Topic-Logistic%20Regression%20%26%20Classification-orange?style=for-the-badge of Contents**
1. [Introduction to Classification](#introduction-to-classification)
2. [Logistic Regression Fundamentals](#logistic-regression-fundamentals)
3. [Activation Functions](#activation-functions)
4. [Hypothesis Representation](#hypothesis-representation)
5. [Decision Boundary](#decision-boundary)
6. [Cost Function for Logistic Regression](#cost-function-for-logistic-regression)
7. [Why MSE Fails in Logistic Regression](#why-mse-fails-in-logistic-regression)
8. [Gradient Descent for Logistic Regression](#gradient-descent-for-logistic-regression)
9. [Overfitting and Underfitting](#overfitting-and-underfitting)
10. [Regularization](#regularization)
11. [Regularized Linear Regression](#regularized-linear-regression)
12. [Regularized Logistic Regression](#regularized-logistic-regression)
13. [Normal Equation](#normal-equation)
14. [Multiclass Classification](#multiclass-classification)
15. [Complete Activation Functions Reference](#complete-activation-functions-reference)

***

## **Introduction to Classification**

### **What is Classification?**

**Definition:** Classification is a supervised learning task where the goal is to predict discrete categorical labels (classes) rather than continuous values.[1]

**Explanation:** Unlike regression, where you predict numeric values (like house prices), classification predicts which category something belongs to. The output variable (y) can take only a limited number of values, typically representing different classes or categories. When you know the correct answers for training data and the outputs are discrete, you're dealing with supervised classification.[1]

**Real-World Examples :**[1]

**Binary Classification (Two Classes):**
- **Email Filtering:** Spam / Not Spam?
- **Fraud Detection:** Fraudulent transaction (Yes / No)?
- **Loan Approval:** Should a bank give a loan to a person or NOT?
- **Medical Diagnosis:** Tumor: Malignant (1) / Benign (0)?
- **Student Admission:** Should a student be admitted to a school or not?
- **Marketing:** Will customers subscribe to a magazine?

**Multiclass Classification (More Than Two Classes):**
- **Voting Behavior:** Which people are more likely to vote for which candidate?
- **Customer Behavior:** Which customers are more likely to buy which product?
- **Image Recognition:** Classify images into multiple categories (cat, dog, bird, etc.)

**Standard Notation :**[1]
- **0:** "Negative Class" (e.g., benign tumor, not spam)
- **1:** "Positive Class" (e.g., malignant tumor, spam)

---

### **Classification vs. Regression**

**Key Differences:**

| **Aspect** | **Regression** | **Classification** |
|------------|----------------|-------------------|
| **Output Type** | Continuous numerical values | Discrete categorical labels |
| **Example Output** | $145,000 (house price) | Spam/Not Spam |
| **Prediction Range** | Any real number | Fixed set of classes |
| **Algorithms** | Linear Regression, Polynomial Regression | Logistic Regression, Decision Trees, SVM |

***

## **Logistic Regression Fundamentals**

### **Definition**

**Definition:** Logistic Regression is a classification algorithm where the dependent variable (y) is categorical and can take only a limited number of values. When there are only two possible outcomes, it's called **Binary Logistic Regression**.[1]

**Explanation:** Despite its name containing "regression," logistic regression is actually a **classification** algorithm. The key difference from linear regression is that the dependent variable is not continuous—it's categorical (e.g., 0 or 1, Yes or No, True or False).[1]

***

### **Why Not Use Linear Regression for Classification?**

**Problem:** If we apply linear regression directly to a classification problem, several issues arise :[1]

1. **Predictions Outside  Range:**[2]
   - For classification, we want predictions to represent probabilities: $$0 \leq p \leq 1$$
   - Linear regression hypothesis $$h_\theta(x)$$ can produce values $$> 1$$ or $$< 0$$
   - Example: $$h_\theta(x) = 3.5$$ or $$h_\theta(x) = -2.1$$ makes no sense as a probability

2. **Unbounded Output:**
   - The weighted sum of inputs in linear regression: $$h_\theta(x) = \theta^T x$$
   - This value can be any real number from $$-\infty$$ to $$+\infty$$
   - We need a mechanism to constrain output between 0 and 1[1]

3. **Threshold Ambiguity:**
   - Setting a threshold like 0.5 for classification is arbitrary
   - Outliers can drastically shift the linear regression line
   - This makes the model unreliable for classification tasks[1]

**Visual Example :**[1]

Consider predicting tumor malignancy based on tumor size:
- **Tumor Size** (x-axis) vs. **Malignant?** (y-axis: 0=No, 1=Yes)
- A linear regression line might fit training data but extends beyond[2]
- A new large tumor could push the line, making small tumors incorrectly classified

***

### **Solution: Bounded Output Using Activation Functions**

**Approach:** Instead of directly outputting the weighted sum $$\theta^T x$$, we pass it through a function that maps any real value to the range.[2][1]

**Mathematical Framework :**[1]

1. **Compute Linear Combination:**
   $$
   z = h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n
   $$

2. **Apply Activation Function:**
   $$
   g(z) = g(h_\theta(x)) = g(\theta^T x)
   $$
   where $$g$$ is an activation function that constrains the output

3. **Final Hypothesis:**
   $$
   h_\theta(x) = g(\theta^T x)
   $$

***

## **Activation Functions**

### **What is an Activation Function?**

**Definition:** An activation function is a mathematical function that transforms the weighted sum of inputs into a bounded or non-linear output, enabling neural networks and machine learning models to learn complex patterns.[1]

**Explanation:** The activation function is applied after computing the linear combination of inputs. Think of it as a "filter" or "gate" that decides what output should be passed forward based on the input. Different activation functions serve different purposes—some bound outputs to ranges, others introduce non-linearity.[1]

**General Form :**[1]
$$
\text{Output} = g(h_\theta(x)) = g(\theta^T x)
$$
where $$g$$ is the activation function.

***

### **1. Linear Activation Function**

**Definition:** The linear activation function returns the input without any transformation.[1]

**Mathematical Formula :**[1]
$$
g(\theta^T x) = \theta^T x
$$

**Explanation:** This activation function does nothing—it's essentially the identity function. The output equals the input, which means the neuron performs only a linear transformation. This is similar to the equation of a straight line: $$y = mx + b$$.[1]

**Output Range:** $$-\infty$$ to $$+\infty$$[1]

**Use Cases :**[1]
- **Simple regression problems** where you want to predict continuous values
- **Output layer** in regression tasks (e.g., housing price prediction)
- **Not suitable** for classification or when non-linearity is needed

**Characteristics:**
- **Range:** Unbounded ($$-\infty, +\infty$$)
- **Derivative:** Constant (doesn't depend on input)
- **Non-linearity:** None (purely linear)

**Advantages:**
- Computationally simple
- No saturation (derivative never becomes zero)

**Disadvantages :**[1]
- **The derivative is constant**, meaning gradient doesn't depend on input—this limits learning
- **No non-linearity** means stacking multiple layers with linear activations is equivalent to a single layer
- **Cannot solve complex problems** that require non-linear decision boundaries

**Example:**
- Input: $$z = 3.5$$
- Output: $$g(z) = 3.5$$ (unchanged)

***

### **2. Sigmoid Activation Function**

**Definition:** The sigmoid (logistic) function is an S-shaped curve that maps any real-valued number to a value between 0 and 1.[1]

**Mathematical Formula :**[1]
$$
g(z) = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

Alternative notation:
$$
g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
$$

where:
- $$z = \theta^T x = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots$$
- $$e \approx 2.718$$ (base of natural logarithm)

**Explanation:** The sigmoid function "squashes" any input value into the range (0, 1). Large positive values approach 1, large negative values approach 0, and the midpoint (z=0) maps to 0.5. This makes it perfect for binary classification, as the output can be interpreted as a probability.[1]

**Output Range:** $$(0, 1)$$ — strictly between 0 and 1 (never exactly 0 or 1)[1]

**Key Properties :**[1]

1. **At z = 0:**
   $$
   g(0) = \frac{1}{1 + e^0} = \frac{1}{2} = 0.5
   $$

2. **As z → +∞:**
   $$
   g(z) \rightarrow 1
   $$

3. **As z → -∞:**
   $$
   g(z) \rightarrow 0
   $$

4. **Smooth S-shaped curve:**
   - Gradually transitions from 0 to 1
   - Symmetric around z = 0

**Python Implementation :**[1]
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

**Use Cases:**
- **Binary classification** (the most common activation for logistic regression)
- **Output layer** when predicting probabilities
- **Hidden layers** in shallow neural networks (though less common now due to vanishing gradient)

**Advantages:**
- **Output interpretable as probability:** Values between 0 and 1
- **Smooth gradient:** Differentiable everywhere
- **Clear probabilistic interpretation:** Easy to understand predictions

**Disadvantages:**
- **Vanishing gradient problem:** For very large or very small z, gradient becomes extremely small
- **Output not zero-centered:** All outputs are positive, which can slow learning
- **Computationally expensive:** Involves exponential function

**Example Values:**
- $$z = -5$$: $$g(-5) \approx 0.0067$$ (close to 0)
- $$z = 0$$: $$g(0) = 0.5$$ (midpoint)
- $$z = 5$$: $$g(5) \approx 0.9933$$ (close to 1)

**Visualization:** The sigmoid function creates an S-curve with:
- Horizontal asymptote at y = 0 (bottom)
- Horizontal asymptote at y = 1 (top)
- Inflection point at (0, 0.5)

***

### **3. Rectified Linear Activation Function (ReLU)**

**Definition:** ReLU (Rectified Linear Unit) outputs the input directly if it's positive; otherwise, it outputs zero.[1]

**Mathematical Formula :**[1]
$$
g(z) = \text{ReLU}(z) = \max(0, z) = \begin{cases}
z & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
$$

**Explanation:** ReLU is very simple—it acts as a "gate" that only allows positive values to pass through. Negative values are blocked and replaced with zero. This introduces non-linearity while being computationally efficient.[1]

**Output Range:** $$[0, +\infty)$$[1]

**Use Cases :**[1]
- **Hidden layers** in deep neural networks (most popular choice)
- **Convolutional Neural Networks** (CNNs)
- **General-purpose deep learning** (default choice for many architectures)

**Advantages:**
- **Computationally efficient:** No exponential calculations
- **Reduces vanishing gradient:** Gradient is either 0 or 1
- **Sparse activation:** Many neurons output 0, creating sparse representations
- **Faster convergence:** Typically trains faster than sigmoid/tanh

**Disadvantages:**
- **Dying ReLU problem:** Neurons can "die" if they output 0 for all inputs (gradient becomes 0)
- **Not zero-centered:** All outputs are non-negative
- **Unbounded output:** Can lead to exploding activations

**Example Values:**
- $$z = -3$$: $$g(-3) = 0$$
- $$z = 0$$: $$g(0) = 0$$
- $$z = 5$$: $$g(5) = 5$$

**Variants:**

#### **Leaky ReLU**

**Definition:** A variant of ReLU that allows a small gradient for negative values.[1]

**Mathematical Formula :**[1]
$$
g(z) = \begin{cases}
z & \text{if } z > 0 \\
0.01z & \text{if } z \leq 0
\end{cases}
$$

**Explanation:** Instead of completely blocking negative values, Leaky ReLU allows a small, constant gradient (typically 0.01) for negative inputs. This helps prevent the "dying ReLU" problem.[1]

**Example Values:**
- $$z = -10$$: $$g(-10) = -0.1$$ (small negative value preserved)
- $$z = 5$$: $$g(5) = 5$$ (same as ReLU for positives)

---

### **4. Hyperbolic Tangent (tanh) Activation Function**

**Definition:** The hyperbolic tangent function maps input values to a range between -1 and 1.[1]

**Mathematical Formula :**[1]
$$
g(z) = \tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}} = \frac{e^{2z} - 1}{e^{2z} + 1}
$$

**Explanation:** Tanh is similar to the sigmoid function but centered at zero, meaning its output range is (-1, 1) instead of (0, 1). This zero-centering property often makes it preferable to sigmoid in hidden layers, as it helps with gradient flow during backpropagation.[1]

**Output Range:** $$(-1, 1)$$[1]

**Key Properties :**[1]

1. **At z = 0:**
   $$
   \tanh(0) = 0
   $$

2. **As z → +∞:**
   $$
   \tanh(z) \rightarrow 1
   $$

3. **As z → -∞:**
   $$
   \tanh(z) \rightarrow -1
   $$

4. **Zero-centered:** Unlike sigmoid, outputs are symmetric around 0

**Relationship to Sigmoid:**
$$
\tanh(z) = 2\sigma(2z) - 1
$$
(Tanh is a scaled and shifted version of sigmoid)

**Use Cases:**
- **Hidden layers** in neural networks (better than sigmoid)
- **RNNs and LSTMs** (commonly used in sequence models)
- **When zero-centered outputs are beneficial**

**Advantages:**
- **Zero-centered output:** Better gradient flow than sigmoid
- **Stronger gradients:** Derivative is steeper than sigmoid
- **Symmetric around origin:** Outputs are balanced

**Disadvantages:**
- **Vanishing gradient problem:** Similar to sigmoid for extreme values
- **Computationally expensive:** Involves exponential calculations

**Example Values:**
- $$z = -5$$: $$\tanh(-5) \approx -0.9999$$
- $$z = 0$$: $$\tanh(0) = 0$$
- $$z = 5$$: $$\tanh(5) \approx 0.9999$$

***

### **5. Softmax Activation Function**

**Definition:** Softmax is a multi-class generalization of the sigmoid function that converts a vector of raw scores (logits) into a probability distribution.[1]

**Mathematical Formula :**[1]

For a vector of K class scores $$z = [z_1, z_2, \ldots, z_K]$$:
$$
g(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Explanation:** Softmax takes a vector of any real values and transforms them into probabilities that sum to 1. Each output represents the probability of the input belonging to a particular class. The class with the highest probability is typically chosen as the prediction.[1]

**Key Properties :**[1]

1. **All outputs are positive:** $$0 < g(z_i) < 1$$
2. **Outputs sum to 1:** $$\sum_{i=1}^{K} g(z_i) = 1$$
3. **Maintains relative ordering:** Larger input values get larger probabilities
4. **Differentiable:** Smooth gradients for backpropagation

**Computation Steps :**[1]

**Step 1:** Exponentiate every element of the output layer
$$
e^{z_1}, e^{z_2}, \ldots, e^{z_K}
$$

**Step 2:** Sum all the exponentials
$$
S = \sum_{j=1}^{K} e^{z_j}
$$

**Step 3:** Divide each exponential by the sum
$$
p_i = \frac{e^{z_i}}{S}
$$

**Example Calculation :**[1]

**Input scores:** $$z = [1.3, 5.1, 2.2, 0.7]$$

**Step 1: Exponentiate**
- $$e^{1.3} \approx 3.67$$
- $$e^{5.1} \approx 164.02$$
- $$e^{2.2} \approx 9.03$$
- $$e^{0.7} \approx 2.01$$

**Step 2: Sum**
$$
S = 3.67 + 164.02 + 9.03 + 2.01 = 178.73
$$

**Step 3: Compute probabilities**
- $$p_1 = \frac{3.67}{178.73} \approx 0.02$$ (2%)
- $$p_2 = \frac{164.02}{178.73} \approx 0.92$$ (92%)
- $$p_3 = \frac{9.03}{178.73} \approx 0.05$$ (5%)
- $$p_4 = \frac{2.01}{178.73} \approx 0.01$$ (1%)

**Predicted class:** Class 2 (highest probability: 92%)

**Use Cases :**[1]
- **Multiclass classification** (more than 2 classes)
- **Output layer** in neural networks for classification
- **Mutually exclusive classes** (each sample belongs to exactly one class)

**Advantages:**
- **Probabilistic interpretation:** Clear probability for each class
- **Differentiable:** Works well with gradient descent
- **Handles multiple classes:** Natural extension of binary classification

**Disadvantages:**
- **Computationally expensive:** Requires exponentials for all classes
- **Sensitive to outliers:** Very large logits can dominate
- **Not suitable for multi-label:** Assumes mutually exclusive classes

**When to Use:**
- Use **Sigmoid** for binary classification or multi-label classification
- Use **Softmax** for multiclass classification (exactly one class per sample)

---

### **Activation Function Selection Guide**

| **Layer Type** | **Recommended Activation** | **Why?** |
|----------------|---------------------------|----------|
| **Hidden Layers (Deep Learning)** | ReLU or Leaky ReLU | Fast, reduces vanishing gradient |
| **Hidden Layers (Shallow Networks)** | tanh | Zero-centered, stronger gradients than sigmoid |
| **Binary Classification Output** | Sigmoid | Outputs probability between 0 and 1 |
| **Multiclass Classification Output** | Softmax | Outputs probability distribution over classes |
| **Regression Output** | Linear | Allows any real-valued output |

***

## **Hypothesis Representation**

### **Logistic Regression Hypothesis**

**Definition:** The hypothesis function in logistic regression outputs the probability that the input belongs to the positive class (y=1), using the sigmoid activation function.[1]

**Mathematical Formula :**[1]
$$
h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
$$

where:
- $$g(z)$$ is the sigmoid function
- $$z = \theta^T x = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$
- $$\theta$$ are the parameters (weights)
- $$x$$ are the features

**Constraint:** We want $$0 \leq h_\theta(x) \leq 1$$[1]

***

### **Interpretation of Hypothesis Output**

**Definition:** The output $$h_\theta(x)$$ represents the estimated probability that y = 1 given input x, parameterized by θ.[1]

**Mathematical Notation :**[1]
$$
h_\theta(x) = P(y=1 \mid x; \theta)
$$

Read as: "Probability that y equals 1, given x, parameterized by theta"

**Explanation:** The hypothesis doesn't directly predict 0 or 1. Instead, it gives you the probability of the positive class. You interpret this probability to make a decision. For example, if $$h_\theta(x) = 0.7$$, there's a 70% chance that y = 1.[1]

**Practical Example :**[1]

**Medical Diagnosis Scenario:**
- **Problem:** Predicting if a tumor is malignant
- **Features:** Tumor size, patient age, etc.
- **Hypothesis output:** $$h_\theta(x) = 0.7$$

**Interpretation:** 
- "There is a 70% chance that the tumor is malignant"
- You would tell the patient: "Based on the data, there's a 70% probability that the tumor is malignant"

**Complementary Probability:**
Since probabilities sum to 1:
$$
P(y=0 \mid x; \theta) = 1 - P(y=1 \mid x; \theta) = 1 - h_\theta(x)
$$

In the tumor example:
- $$P(\text{malignant}) = 0.7$$
- $$P(\text{benign}) = 1 - 0.7 = 0.3$$ (30% chance)

***

## **Decision Boundary**

### **What is a Decision Boundary?**

**Definition:** The decision boundary is the line, curve, or surface that separates different classes in the feature space. It's where the hypothesis equals exactly 0.5 (or where the underlying score equals zero).[1]

**Explanation:** The decision boundary is the "dividing line" between regions where we predict y=1 and regions where we predict y=0. It's determined by the parameters θ and the structure of the hypothesis function.[1]

***

### **Decision Rule**

**Standard Threshold at 0.5 :**[1]

**Predict y = 1 if:**
$$
h_\theta(x) \geq 0.5
$$

**Predict y = 0 if:**
$$
h_\theta(x) < 0.5
$$

**Explanation:** We typically use 0.5 as the decision threshold. Since sigmoid outputs probabilities, 0.5 means "equally likely to be either class." Above 0.5, we lean toward y=1; below 0.5, we lean toward y=0.[1]

***

### **Relationship to Sigmoid Function**

**Key Insight :**[1]

The sigmoid function $$g(z) = \frac{1}{1 + e^{-z}}$$ has these properties:

1. **When $$z \geq 0$$:**
   $$
   g(z) \geq 0.5
   $$

2. **When $$z < 0$$:**
   $$
   g(z) < 0.5
   $$

**Therefore, since $$z = \theta^T x$$:**

**Predict y = 1 whenever:**
$$
\theta^T x \geq 0
$$

**Predict y = 0 whenever:**
$$
\theta^T x < 0
$$

**Conclusion:** The decision boundary is defined by $$\theta^T x = 0$$.[1]

***

### **Linear Decision Boundary**

**Example :**[1]

**Hypothesis:**
$$
h_\theta(x) = g(\theta_0 + \theta_1 x_1 + \theta_2 x_2)
$$

**Parameters:**
$$
\theta = \begin{bmatrix} -3 \\ 1 \\ 1 \end{bmatrix}
$$

**Decision Boundary Equation:**

Set $$\theta^T x = 0$$:
$$
-3 + x_1 + x_2 = 0
$$
$$
x_2 = 3 - x_1
$$

**Interpretation:** This is a straight line in the $$x_1$$-$$x_2$$ plane with:
- **Slope:** -1
- **y-intercept:** 3

**Prediction Regions:**
- **Above the line** ($$x_1 + x_2 \geq 3$$): Predict y = 1
- **Below the line** ($$x_1 + x_2 < 3$$): Predict y = 0

**Visual Example :**[1]
```
x2 |
 3 |     /
   |    /
 2 |   / (y=1 region)
   |  /
 1 | / (y=0 region)
   |/______________
   0   1   2   3  x1
```

***

### **Non-Linear Decision Boundary**

**Definition:** When polynomial features are used, the decision boundary becomes a curve or complex shape rather than a straight line.[1]

**Example :**[1]

**Hypothesis with Polynomial Features:**
$$
h_\theta(x) = g(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_1^2 + \theta_4 x_2^2)
$$

**Parameters:**
$$
\theta = \begin{bmatrix} -1 \\ 0 \\ 0 \\ 1 \\ 1 \end{bmatrix}
$$

**Decision Boundary Equation:**

Set $$\theta^T x = 0$$:
$$
-1 + x_1^2 + x_2^2 = 0
$$
$$
x_1^2 + x_2^2 = 1
$$

**Interpretation:** This is a **circle** centered at the origin with radius 1!

**Prediction Regions:**
- **Inside the circle** ($$x_1^2 + x_2^2 < 1$$): Predict y = 0
- **Outside the circle** ($$x_1^2 + x_2^2 \geq 1$$): Predict y = 1

**Visual Example :**[1]
```
x2 |
 1 |    ___
   |  /     \
 0 | |   o   | (y=0 inside)
   |  \___  /
-1 |      
   |_____________
  -1   0   1   x1
   
(y=1 outside circle)
```

**More Complex Boundaries:**

By including higher-order polynomial terms, you can create arbitrarily complex decision boundaries:
- Ellipses
- Parabolas
- Irregular shapes

**Example:** 
$$
h_\theta(x) = g(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_1^2 + \theta_4 x_1 x_2 + \theta_5 x_2^2 + \cdots)
$$

***

### **Example Dataset: University Admission**

**Problem Description :**[1]
- **Dataset:** Marks of two exams for 100 university applicants
- **Features:** 
  - $$x_1$$: Score on Exam 1
  - $$x_2$$: Score on Exam 2
- **Target Variable:**
  - y = 1: Applicant was admitted
  - y = 0: Applicant was not admitted

**Goal:** Learn a decision boundary that separates admitted students from rejected students based on their exam scores.

**Visual Representation:**
When plotted, you'd see:
- Blue points (y=0): Rejected students
- Red points (y=1): Admitted students
- Decision boundary: The line/curve separating the two groups

**Interpretation:** Once trained, the model can predict whether a new applicant with exam scores $$(x_1, x_2)$$ will be admitted based on which side of the decision boundary they fall.

***

## **Cost Function for Logistic Regression**

### **Overview**

**Question:** How do we choose the parameters θ for logistic regression?[1]

**Answer:** We need a cost function that measures how well our hypothesis fits the training data. We then minimize this cost function to find optimal parameters.[1]

---

### **Training Set Notation**

**Given :**[1]
- **m examples:** $$\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \ldots, (x^{(m)}, y^{(m)})\}$$
- **n features:** Each $$x^{(i)} \in \mathbb{R}^{n+1}$$ (including $$x_0 = 1$$)
- **Binary labels:** $$y^{(i)} \in \{0, 1\}$$

**Hypothesis:**
$$
h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}
$$

***

### **Why Not Use Mean Squared Error (MSE)?**

**Linear Regression Cost Function :**[1]

For linear regression, we use Mean Squared Error:
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
$$

**Generic Form:**
$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \text{Cost}(h_\theta(x^{(i)}), y^{(i)})
$$

where:
$$
\text{Cost}(h_\theta(x), y) = \frac{1}{2}(h_\theta(x) - y)^2
$$

**Problem with MSE for Logistic Regression :**[1]

When we plug the sigmoid hypothesis into MSE:
$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \frac{1}{2}\left(\frac{1}{1 + e^{-\theta^T x^{(i)}}} - y^{(i)}\right)^2
$$

This creates a **non-convex** cost function!

***

## **Why MSE Fails in Logistic Regression**

### **Convex vs. Non-Convex Functions**

**Convex Function (Linear Regression) :**[1]

**Definition:** A function where any line segment between two points on the curve lies above or on the curve. It has a single global minimum.

**Characteristics:**
- **Single bowl shape:** One minimum point
- **Gradient descent guaranteed to converge** to global minimum
- **Reliable optimization:** Always finds the best solution

**Visual:** Imagine a smooth bowl—no matter where you start, rolling down always leads to the bottom.

**Non-Convex Function (Logistic Regression with MSE) :**[1]

**Definition:** A function with multiple local minima and maxima. It has a "wavy" or "bumpy" surface.

**Characteristics:**
- **Multiple local minima:** Many "valleys" in the cost surface
- **Gradient descent can get stuck** in local minima (not the global minimum)
- **Unreliable optimization:** Might not find the best solution

**Visual:** Imagine a mountain range with multiple valleys—you might get stuck in a small valley instead of reaching the deepest one.

***

### **Why Does This Happen?**

**Root Cause :**[1]

The non-convexity arises from the **sigmoid function** $$g(z) = \frac{1}{1 + e^{-z}}$$ inside the squared error term.

**Mathematical Explanation:**
- The sigmoid function is **non-linear**
- Squaring a non-linear function creates **complex curves** with multiple extrema
- The composition of exponential and quadratic functions produces a non-convex surface

**Impact on Optimization:**
- Gradient descent might converge to different solutions depending on initialization
- No guarantee of finding the globally best parameters
- Training becomes unreliable and inconsistent

***

### **Solution: Log Loss (Cross-Entropy) Cost Function**

**Definition:** The log loss (also called cross-entropy loss or binary cross-entropy) is specifically designed for binary classification with logistic regression.[1]

**Cost Function :**[1]
$$
\text{Cost}(h_\theta(x), y) = \begin{cases}
-\log(h_\theta(x)) & \text{if } y = 1 \\
-\log(1 - h_\theta(x)) & \text{if } y = 0
\end{cases}
$$

**Explanation:** The cost depends on the true label y. We use different formulas for y=1 and y=0, both involving logarithms. This creates a convex cost function suitable for optimization.[1]

---

### **Understanding the Cost Function: Case y = 1**

**When the true label is y = 1 :**[1]
$$
\text{Cost}(h_\theta(x), 1) = -\log(h_\theta(x))
$$

**Behavior:**

1. **If $$h_\theta(x) = 1$$ (perfect prediction):**
   $$
   \text{Cost} = -\log(1) = 0
   $$
   - **No penalty:** Model predicted correctly with full confidence

2. **If $$h_\theta(x) \rightarrow 0$$ (completely wrong):**
   $$
   \text{Cost} = -\log(0) \rightarrow \infty
   $$
   - **Infinite penalty:** Model predicted the opposite with full confidence

3. **If $$h_\theta(x) = 0.5$$ (uncertain):**
   $$
   \text{Cost} = -\log(0.5) \approx 0.69
   $$
   - **Moderate penalty:** Model is unsure

**Visual Shape:**
```
Cost
  |
∞ |                      
  |                      
  |                      
  |\                     
  | \                    
  |  \___                
  |      ----____        
0 |____________----____
  0    0.5           1   h_θ(x)
```

**Interpretation:** As the prediction moves away from 1 (the true label), the cost increases exponentially. This strongly penalizes confident wrong predictions.[1]

***

### **Understanding the Cost Function: Case y = 0**

**When the true label is y = 0 :**[1]
$$
\text{Cost}(h_\theta(x), 0) = -\log(1 - h_\theta(x))
$$

**Behavior:**

1. **If $$h_\theta(x) = 0$$ (perfect prediction):**
   $$
   \text{Cost} = -\log(1 - 0) = -\log(1) = 0
   $$
   - **No penalty:** Model predicted correctly with full confidence

2. **If $$h_\theta(x) \rightarrow 1$$ (completely wrong):**
   $$
   \text{Cost} = -\log(1 - 1) = -\log(0) \rightarrow \infty
   $$
   - **Infinite penalty:** Model predicted the opposite with full confidence

3. **If $$h_\theta(x) = 0.5$$ (uncertain):**
   $$
   \text{Cost} = -\log(0.5) \approx 0.69
   $$
   - **Moderate penalty:** Model is unsure

**Visual Shape:**
```
Cost
  |
∞ |                      
  |                     /
  |                   /  
  |                 /    
  |              __/     
  |         ____/        
  |   _____/             
0 |--/________________
  0    0.5           1   h_θ(x)
```

**Interpretation:** As the prediction moves away from 0 (the true label), the cost increases exponentially. This mirrors the y=1 case but flipped.[1]

***

### **Simplified Cost Function**

**Combined Formula :**[1]

Instead of writing two separate cases, we can combine them using mathematical elegance:
$$
\text{Cost}(h_\theta(x), y) = -y \log(h_\theta(x)) - (1-y) \log(1 - h_\theta(x))
$$

**Why This Works:**

1. **When y = 1:**
   $$
   \text{Cost} = -1 \cdot \log(h_\theta(x)) - (1-1) \cdot \log(1 - h_\theta(x)) = -\log(h_\theta(x))
   $$

2. **When y = 0:**
   $$
   \text{Cost} = -0 \cdot \log(h_\theta(x)) - (1-0) \cdot \log(1 - h_\theta(x)) = -\log(1 - h_\theta(x))
   $$

**Complete Cost Function :**[1]

For all m training examples:
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
$$

**Explanation:** This is the average cost across all training examples. The negative sign ensures that we're minimizing cost (since log values are negative for probabilities less than 1).[1]

**Key Properties:**
- **Convex function:** Guaranteed single global minimum
- **Differentiable:** Smooth gradients for optimization
- **Derived from maximum likelihood:** Has strong statistical foundations

***

### **Making Predictions**

**To Fit Parameters θ :**[1]
$$
\min_\theta J(\theta)
$$

Use gradient descent or other optimization algorithms to find θ that minimizes the cost.

**To Make a Prediction on New Input x :**[1]
$$
h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}
$$

**Decision Rule:**
- If $$h_\theta(x) \geq 0.5$$: Predict y = 1
- If $$h_\theta(x) < 0.5$$: Predict y = 0

***

## **Gradient Descent for Logistic Regression**

### **Algorithm**

**Goal:** Minimize $$J(\theta)$$[1]

**Gradient Descent Update Rule :**[1]

Repeat until convergence:
$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
$$

Simultaneously update all $$\theta_j$$ for $$j = 0, 1, 2, \ldots, n$$.

Where:
- $$\alpha$$ is the learning rate
- $$\frac{\partial}{\partial \theta_j} J(\theta)$$ is the partial derivative (gradient)

---

### **Gradient Computation**

**Derivative of the Cost Function :**[1]
$$
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

**Complete Update Rule :**[1]

Repeat:
$$
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

(Simultaneously update all $$\theta_j$$)

***

### **Observation: Similar to Linear Regression**

**Remarkable Fact :**[1]

The gradient descent algorithm **looks identical** to linear regression!

**Linear Regression Update:**
$$
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

**Logistic Regression Update:**
$$
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

**Key Difference:**

The formulas look the same, but **the hypothesis function is different:**

- **Linear Regression:** $$h_\theta(x) = \theta^T x$$
- **Logistic Regression:** $$h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$$

This means the gradients are computed differently even though the update formula has the same structure.[1]

---

### **Complete Derivative of Logistic Cost Function**

**Derivation Steps :**[1]

Starting from:
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
$$

**Step 1: Apply Chain Rule**

Let $$h = h_\theta(x^{(i)})$$ and $$z = \theta^T x^{(i)}$$.

**Step 2: Derivative of Sigmoid**

Key property:
$$
\frac{dg(z)}{dz} = g(z)(1 - g(z))
$$

where $$g(z) = \frac{1}{1 + e^{-z}}$$

**Step 3: Apply to Cost Function**

Through calculus (chain rule and logarithmic differentiation):
$$
\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

**Intuition:** The gradient points in the direction of steepest increase in cost. By moving in the opposite direction (subtracting the gradient), we move toward lower cost.[1]

***

### **Vectorized Implementation**

**Vectorized Gradient :**[1]
$$
\nabla J(\theta) = \frac{1}{m} X^T (h - y)
$$

where:
- $$X$$ is the $$m \times (n+1)$$ design matrix
- $$h = g(X\theta)$$ is the vector of all predictions
- $$y$$ is the vector of true labels

**Vectorized Update:**
$$
\theta := \theta - \alpha \nabla J(\theta) = \theta - \frac{\alpha}{m} X^T (g(X\theta) - y)
$$

**Advantage:** Much faster computation using matrix operations instead of loops.

***

## **Overfitting and Underfitting**

### **The Problem of Overfitting**

**Definition:** Overfitting occurs when a model learns the training data too well, including its noise and outliers, resulting in poor generalization to new, unseen data.[1]

**Explanation:** Imagine memorizing answers to practice problems without understanding the concepts. You'd ace the practice test but fail on new problems. Similarly, an overfitted model "memorizes" training data patterns (including random noise) rather than learning the underlying true patterns. It performs excellently on training data but poorly on test data.[1]

***

### **Visualizing Overfitting in Regression**

**Three Scenarios for Housing Price Prediction :**[1]

#### **1. Underfitting (High Bias)**

**Model:** Linear function
$$
h_\theta(x) = \theta_0 + \theta_1 x
$$

**Characteristics:**
- **Too simple:** Straight line trying to fit curved data
- **High training error:** Doesn't fit training data well
- **High test error:** Also doesn't fit new data well
- **Problem:** Model is too simple to capture the underlying pattern

**Visual:** A straight line through obviously curved data points—clearly inadequate.

**Also called:** "High bias" because the model has a strong preconception (bias) that the relationship is linear.

---

#### **2. Just Right (Good Fit)**

**Model:** Quadratic function
$$
h_\theta(x) = \theta_0 + \theta_1 x + \theta_2 x^2
$$

**Characteristics:**
- **Appropriate complexity:** Captures the main trend without overfitting
- **Low training error:** Fits training data reasonably well
- **Low test error:** Generalizes well to new data
- **Sweet spot:** Balances bias and variance

**Visual:** A smooth curve that follows the general trend of the data without trying to hit every point exactly.

***

#### **3. Overfitting (High Variance)**

**Model:** High-degree polynomial
$$
h_\theta(x) = \theta_0 + \theta_1 x + \theta_2 x^2 + \theta_3 x^3 + \theta_4 x^4 + \cdots
$$

**Characteristics:**
- **Too complex:** Wiggly curve that passes through (or very close to) every training point
- **Very low training error:** $$J(\theta) = \frac{1}{2m} \sum (h_\theta(x^{(i)}) - y^{(i)})^2 \approx 0$$
- **High test error:** Fails to generalize to new examples
- **Problem:** Model captures noise instead of signal

**Visual:** A wildly oscillating curve that threads through all training points but makes unreasonable predictions between them.[1]

**Also called:** "High variance" because the model's predictions vary wildly with small changes in training data.

***

### **Visualizing Overfitting in Classification**

**Three Scenarios for Logistic Regression :**[1]

#### **1. Underfitting (High Bias)**

**Model:** Simple linear boundary
$$
h_\theta(x) = g(\theta_0 + \theta_1 x_1 + \theta_2 x_2)
$$

**Characteristics:**
- **Straight line decision boundary**
- **Many misclassifications** in training data
- **Too simple** to capture the actual class separation

**Visual:** A straight line trying to separate two classes that clearly need a curved boundary.

***

#### **2. Just Right (Good Fit)**

**Model:** Moderately complex boundary
$$
h_\theta(x) = g(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_1^2 + \theta_4 x_2^2)
$$

**Characteristics:**
- **Curved decision boundary** (e.g., ellipse or smooth curve)
- **Good separation** of classes
- **Generalizes well** to new data

**Visual:** A smooth curve that effectively separates the two classes without being overly complex.

***

#### **3. Overfitting (High Variance)**

**Model:** Very complex boundary
$$
h_\theta(x) = g(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_1^2 + \theta_4 x_1 x_2 + \theta_5 x_2^2 + \cdots)
$$

Where $$g(z) = \frac{1}{1 + e^{-z}}$$ is the sigmoid function.

**Characteristics:**
- **Extremely wiggly decision boundary**
- **Perfect classification** of training data
- **Absurd boundaries** that wrap around individual points
- **Poor generalization** to new data

**Visual:** A convoluted boundary with loops and curves that perfectly classifies every training point but makes no logical sense.[1]

---

### **How to Detect Overfitting**

**Challenge:** With many features, you can't easily plot the hypothesis function.[1]

**Example :**[1]

**Housing price prediction with many features:**
- Size of house
- Number of bedrooms
- Number of floors
- Age of house
- Average income in neighborhood
- Kitchen size
- ... (many more)

**Problem:** You can't visualize a function in high-dimensional space!

**Solution:** Use other diagnostic techniques:
- **Training vs. validation error:** Large gap indicates overfitting
- **Learning curves:** Plot error vs. training set size
- **Cross-validation:** Test on held-out data

***

### **Addressing Overfitting**

**Two Main Approaches :**[1]

***

#### **Option 1: Reduce Number of Features**

**Approaches:**

1. **Manual Feature Selection:**
   - Use domain knowledge to choose which features to keep
   - Remove features that seem irrelevant
   - **Example:** In housing prices, "owner's favorite color" is likely irrelevant

2. **Automated Model Selection Algorithm:**
   - Systematically evaluate different feature subsets
   - Use techniques like forward selection, backward elimination, or wrapper methods
   - Will be covered later in the course[1]

**Advantages:**
- Simpler model
- Faster training and prediction
- Easier to interpret

**Disadvantages:**
- Might discard useful information
- Requires domain expertise or computational search

***

#### **Option 2: Regularization**

**Definition:** Regularization adds a penalty term to the cost function that discourages large parameter values, effectively reducing model complexity while keeping all features.[1]

**Key Idea:**
- Keep all features
- Reduce the magnitude (values) of parameters $$\theta_j$$
- This "shrinks" the influence of less important features
- Works well when many features each contribute a small amount to predicting y[1]

**Advantages:**
- Retain all features (no information loss)
- Automatic feature weighting
- Proven mathematical framework
- Can tune the amount of regularization

**Disadvantages:**
- Adds hyperparameter to tune
- Doesn't provide feature interpretability

***

## **Regularization**

### **Intuition Behind Regularization**

**Motivating Example :**[1]

**Suppose we have:**
$$
h_\theta(x) = \theta_0 + \theta_1 x + \theta_2 x^2 + \theta_3 x^3 + \theta_4 x^4
$$

This fourth-degree polynomial might be overfitting.

**Idea:** What if we penalized $$\theta_3$$ and $$\theta_4$$ to make them really small?

**Modified Cost Function:**
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + 1000 \theta_3^2 + 1000 \theta_4^2
$$

**Effect:**
- To minimize $$J(\theta)$$, the optimization algorithm will make $$\theta_3$$ and $$\theta_4$$ very small (close to zero)
- This effectively reduces the model to approximately quadratic: $$h_\theta(x) \approx \theta_0 + \theta_1 x + \theta_2 x^2$$
- The decision boundary becomes smoother and simpler[1]

**Visual Result :**[1]
- Original: Wiggly overfitted curve
- After penalizing $$\theta_3, \theta_4$$: Smooth quadratic curve (just right)

***

### **General Regularization Framework**

**Problem:** In practice, we don't know in advance which parameters to penalize.

**Solution:** Penalize **all** parameters (except $$\theta_0$$).[1]

**Regularized Cost Function :**[1]
$$
J(\theta) = \frac{1}{2m} \left[ \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2 \right]
$$

**Two Objectives:**

1. **First Term (Data Fitting):**
   $$
   \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
   $$
   - Measures how well the model fits the training data
   - Lower values mean better fit

2. **Second Term (Regularization):**
   $$
   \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
   $$
   - Penalizes large parameter values
   - Encourages simpler model (smoother hypothesis)
   - Note: We typically **don't penalize $$\theta_0$$** (the bias term)[1]

***

### **The Regularization Parameter λ**

**Definition:** $$\lambda$$ (lambda) is the regularization parameter that controls the trade-off between fitting the training data and keeping parameters small.[1]

**Role of λ:**

**Small λ (e.g., λ = 0):**
- Weak regularization
- Model focuses on fitting training data
- Risk of **overfitting**

**Moderate λ:**
- Balanced trade-off
- Good fit with reasonable parameter values
- **Optimal performance** on new data

**Large λ (e.g., λ = 10,000):**
- Strong regularization
- All parameters forced close to zero
- Model becomes too simple
- Risk of **underfitting**[1]

---

### **What Happens with Extremely Large λ?**

**Scenario :**[1]

If $$\lambda$$ is set to an extremely large value (say $$\lambda = 10^{10}$$):

**Effect on Parameters:**
$$
\theta_1 \approx \theta_2 \approx \theta_3 \approx \cdots \approx \theta_n \approx 0
$$

**Resulting Hypothesis:**
$$
h_\theta(x) \approx \theta_0
$$

This is just a **horizontal line**!

**Consequence:**
- The model becomes a constant function
- It predicts the same value for all inputs
- **Severe underfitting:** Fails to capture any patterns in the data[1]

**Visual Example :**[1]
- Data: House prices increasing with size
- Model with huge λ: Horizontal line at average price
- Clearly inadequate!

***

### **Small Values for Parameters**

**Principle :**[1]

By keeping parameter values small through regularization:

1. **"Simpler" Hypothesis:**
   - Smoother decision boundaries
   - Less complex functions
   - Reduced model capacity

2. **Less Prone to Overfitting:**
   - Can't fit to noise as easily
   - Better generalization
   - More robust to variations in training data

**Example: Housing Price Prediction :**[1]

**Features:** $$x_1, x_2, \ldots, x_{100}$$ (100 features)

**Parameters:** $$\theta_0, \theta_1, \theta_2, \ldots, \theta_{100}$$

**Without Regularization:**
- Some $$\theta_j$$ might become very large
- Model might rely heavily on a few features
- Susceptible to overfitting

**With Regularization:**
- All $$\theta_j$$ kept reasonably small
- Model distributes "attention" across features
- More stable and generalizable

***

## **Regularized Linear Regression**

### **Cost Function Formulation**

**Regularized Linear Regression Cost Function :**[1]
$$
J(\theta) = \frac{1}{2m} \left[ \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2 \right]
$$

where:
- $$h_\theta(x) = \theta^T x$$ (linear hypothesis)
- $$m$$ is the number of training examples
- $$n$$ is the number of features
- $$\lambda$$ is the regularization parameter
- We sum from $$j=1$$ to $$n$$ (not including $$\theta_0$$)[1]

***

### **Gradient Descent for Regularized Linear Regression**

**Update Rule :**[1]

Repeat until convergence:

**For $$j = 0$$:**
$$
\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_0^{(i)}
$$

**For $$j = 1, 2, \ldots, n$$:**
$$
\theta_j := \theta_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} \theta_j \right]
$$

**Explanation:** The bias term $$\theta_0$$ is updated without regularization, while all other parameters $$\theta_j$$ (j ≥ 1) include the regularization term $$\frac{\lambda}{m} \theta_j$$.[1]

***

### **Alternative Form: Shrinkage Interpretation**

**Rearranging the update rule for $$j \geq 1$$ :**[1]
$$
\theta_j := \theta_j - \alpha \frac{\lambda}{m} \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

Factor out $$\theta_j$$:
$$
\theta_j := \theta_j \left(1 - \alpha \frac{\lambda}{m}\right) - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

**Interpretation:**

The term $$\left(1 - \alpha \frac{\lambda}{m}\right)$$ is slightly less than 1 (e.g., 0.99).

**Each iteration:**
1. **Shrink** $$\theta_j$$ by multiplying by 0.99
2. **Then** perform the usual gradient descent update

This is why regularization is sometimes called **weight decay**—parameters slowly decay toward zero.[1]

---

## **Normal Equation**

### **Normal Equation for Linear Regression**

**Definition:** The normal equation provides a closed-form (non-iterative) solution to find the optimal parameters for linear regression.[1]

**Standard Normal Equation:**
$$
\theta = (X^T X)^{-1} X^T y
$$

where:
- $$X$$ is the $$m \times (n+1)$$ design matrix
- $$y$$ is the $$m \times 1$$ vector of target values
- $$\theta$$ is the $$(n+1) \times 1$$ parameter vector

***

### **Regularized Normal Equation**

**Regularized Version :**[1]
$$
\theta = \left( X^T X + \lambda \begin{bmatrix}
0 & 0 & 0 & \cdots & 0 \\
0 & 1 & 0 & \cdots & 0 \\
0 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 1
\end{bmatrix} \right)^{-1} X^T y
$$

**Matrix Dimensions:**

The regularization matrix is $$(n+1) \times (n+1)$$ with:
- Top-left element is 0 (no regularization for $$\theta_0$$)
- Remaining diagonal elements are 1 (regularize $$\theta_1, \theta_2, \ldots, \theta_n$$)
- All off-diagonal elements are 0

**Simplified Notation:**

Let $$L$$ be the identity matrix with top-left element changed to 0:
$$
L = \begin{bmatrix}
0 & 0 & 0 & \cdots & 0 \\
0 & 1 & 0 & \cdots & 0 \\
0 & 0 & 1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & 1
\end{bmatrix}
$$

Then:
$$
\theta = (X^T X + \lambda L)^{-1} X^T y
$$

**Advantage:** This regularized version is always invertible (even when $$X^T X$$ alone is not), provided $$\lambda > 0$$.[1]

***

### **MATLAB/Octave Example: Effect of λ**

**Visual Results from Lecture :**[1]

#### **λ = 0 (No Regularization)**
- Highly oscillating curve
- Passes through or very close to all training points
- Clear overfitting
- Poor generalization expected

#### **λ = 1 (Moderate Regularization)**
- Smoother curve
- Balances fit and simplicity
- Better generalization expected
- Reasonable approximation of data

#### **λ = 10 (Strong Regularization)**
- Very smooth curve
- Simpler model
- Might slightly underfit
- Good generalization if underlying relationship is simple

**Takeaway:** The choice of λ is crucial. Cross-validation is typically used to select the optimal value.[1]

***

## **Regularized Logistic Regression**

### **Cost Function**

**Regularized Logistic Regression Cost Function :**[1]
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
$$

where:
- $$h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$$ (logistic hypothesis)
- First term: Log loss (cross-entropy)
- Second term: L2 regularization penalty
- We don't regularize $$\theta_0$$ (sum starts from j=1)[1]

**Visual Example :**[1]

**Without Regularization:**
- Decision boundary might be extremely complex
- Wraps around individual points
- Overfits the training data

**With Regularization:**
- Smoother decision boundary
- More sensible class separation
- Better generalization

***

### **Gradient Descent for Regularized Logistic Regression**

**Update Rule :**[1]

Repeat until convergence:

**For $$j = 0$$:**
$$
\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_0^{(i)}
$$

**For $$j = 1, 2, \ldots, n$$:**
$$
\theta_j := \theta_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} \theta_j \right]
$$

**Explanation:** This looks identical to regularized linear regression, but remember that $$h_\theta(x)$$ is different:
- Linear regression: $$h_\theta(x) = \theta^T x$$
- Logistic regression: $$h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$$
[1]

***

### **Implementation Examples**

**Python with scikit-learn :**[1]

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Convert target variable to binary classification (e.g., > 150)
y_binary = (y > 150).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.25, random_state=42
)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

***

**Python with TensorFlow :**[1]

```python
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Convert target variable to binary classification (e.g., > 150)
y_binary = (y > 150).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
```

***

## **Multiclass Classification**

### **Introduction**

**Definition:** Multiclass classification involves categorizing instances into one of **more than two** classes.[1]

**Binary vs. Multiclass :**[1]

**Binary Classification:**
- Two classes only
- Examples: Spam/Not Spam, Tumor: Malignant/Benign
- Output: y ∈ {0, 1}

**Multiclass Classification:**
- Three or more classes
- Examples: Email categories (Work/Friends/Family/Hobby), Image recognition (Cat/Dog/Bird/Horse)
- Output: y ∈ {1, 2, 3, ..., K}

**Visual Representation :**[1]

**Binary:** Two regions separated by a decision boundary
```
x2 |  ○ ○ ○
   | ○ ○ 
   |____●___●__
   |   ●  ● ●
x1 |  ●
```

**Multiclass:** Multiple regions, each representing a class
```
x2 | △△△ | ○○○
   | △△  | ○○
   |_____|_____
   | ●● | □□□
   | ●●● | □□
x1 |_____|_____
```

***

### **Approaches to Multiclass Classification**

There are several strategies to extend binary classifiers to handle multiple classes.[1]

***

### **1. One-vs-All (One-vs-Rest)**

**Definition:** One-vs-All (OvA), also called One-vs-Rest (OvR), trains K binary classifiers, where each classifier distinguishes one class from all other classes.[1]

**Algorithm :**[1]

For K classes, train K classifiers:

**Classifier 1:**
- Positive examples: Class 1
- Negative examples: Classes 2, 3, ..., K
- Learns: $$h_\theta^{(1)}(x) = P(y=1 \mid x; \theta)$$

**Classifier 2:**
- Positive examples: Class 2
- Negative examples: Classes 1, 3, 4, ..., K
- Learns: $$h_\theta^{(2)}(x) = P(y=2 \mid x; \theta)$$

**Classifier K:**
- Positive examples: Class K
- Negative examples: Classes 1, 2, ..., K-1
- Learns: $$h_\theta^{(K)}(x) = P(y=K \mid x; \theta)$$

**Prediction :**[1]

For a new input $$x$$, compute all K probabilities and pick the class with the highest score:
$$
\text{Predicted class} = \arg\max_{i} h_\theta^{(i)}(x)
$$

**Visual Representation :**[1]

**For 3 classes:**
- Train classifier 1 vs Rest (2+3)
- Train classifier 2 vs Rest (1+3)
- Train classifier 3 vs Rest (1+2)

Each classifier creates a decision boundary separating one class from all others.

***

#### **Architecture Diagram :**[1]

```
         1vR (Classifier 1)
        /
   x → 2vR (Classifier 2) → Winner Takes All → Predicted Class
        \
         3vR (Classifier 3)
        \
         4vR (Classifier 4)
```

**Winner Takes All:** The classifier with the highest confidence "wins."

**Each node is a binary classifier:** Trained to recognize one class vs. all others.[1]

**Prediction Formula:**
$$
\text{Class of } x_i = \arg\max_j h_\theta^{(j)}(x_i)
$$

***

#### **Benefits :**[1]
- **Simple to implement:** Just train M binary classifiers
- **Fast prediction:** Run M classifiers and pick the max
- **Widely supported:** Most machine learning libraries implement this

#### **Drawbacks :**[1]
- **Unbalanced data:** Each classifier sees 1 positive class vs. (M-1) negative classes
  - Example: 4 classes → 1 positive, 3 negatives (75% negative!)
  - Can bias classifiers toward predicting negative
  - **Solutions:** Down-sampling (reduce negative examples) or up-sampling (duplicate positive examples)

***

### **2. One-vs-One (OvO)**

**Definition:** One-vs-One trains a binary classifier for **every pair** of classes.[1]

**Algorithm :**[1]

For K classes, train $$\frac{K(K-1)}{2}$$ classifiers:

**Examples with 4 classes:**
- Classifier: 1 vs 2
- Classifier: 1 vs 3
- Classifier: 1 vs 4
- Classifier: 2 vs 3
- Classifier: 2 vs 4
- Classifier: 3 vs 4

**Total:** $$\frac{4 \times 3}{2} = 6$$ classifiers

**Prediction: Max-Wins Voting :**[1]

1. Run all pairwise classifiers on input $$x$$
2. Each classifier "votes" for one of its two classes
3. Count votes for each class
4. Predict the class with the most votes

**Example:**
- 1 vs 2 → votes for 1
- 1 vs 3 → votes for 3
- 1 vs 4 → votes for 1
- 2 vs 3 → votes for 3
- 2 vs 4 → votes for 4
- 3 vs 4 → votes for 3

**Vote counts:** Class 1: 2, Class 2: 0, Class 3: 3, Class 4: 1

**Prediction:** Class 3 (most votes)

***

#### **Architecture Diagram :**[1]

```
         1v2
        /   \
       /     \
   x → 1v3   2v3 → Max-Wins Voting → Predicted Class
       \     /
        \   /
         2v4, 3v4, 1v4
```

**Max-wins voting:** Count votes from all pairwise classifiers.[1]

---

#### **Benefits:**
- **No unbalanced data:** Each classifier sees exactly 2 classes
- **More focused classifiers:** Each learns one specific distinction
- **Often more accurate** than OvA

#### **Drawbacks:**
- **Many classifiers:** For large K, $$\frac{K(K-1)}{2}$$ grows quickly
  - Example: 10 classes → 45 classifiers
  - Example: 100 classes → 4,950 classifiers!
- **Slower prediction:** Must run many more classifiers

***

### **3. Directed Acyclic Graph (DAG)**

**Definition:** DAG is a hierarchical decision structure that systematically eliminates classes through pairwise comparisons.[1]

**Algorithm :**[1]

**Structure:**
- Construct $$\frac{K(K-1)}{2}$$ binary classifiers (same as OvO)
- Arrange them in a directed acyclic graph (tree-like structure)
- At each node, a pairwise classifier eliminates one class
- Follow the path based on classifier outputs until one class remains

---

## **Continuing Comprehensive Machine Learning Notes**

---

### **3. Directed Acyclic Graph (DAG) - Continued**

**Example with 4 classes :**[21]

```
              Start (All 4 classes)
                      |
                   [1 vs 2]
                   /      \
              Class 1   (2,3,4 remain)
                            |
                        [2 vs 3]
                        /      \
                   Class 2   (3,4 remain)
                                |
                            [3 vs 4]
                            /      \
                       Class 3   Class 4
```

**Algorithm Steps:**

1. **Root Node:** Compare classes 1 vs 2
   - If 1 wins → Predict Class 1 (done)
   - If 2 wins → Continue to next level with classes {2, 3, 4}

2. **Second Level:** Compare classes 2 vs 3
   - If 2 wins → Predict Class 2 (done)
   - If 3 wins → Continue with classes {3, 4}

3. **Final Level:** Compare classes 3 vs 4
   - Winner is the final prediction

**Key Properties:**

**Number of Classifiers:** Still requires $$\frac{K(K-1)}{2}$$ binary classifiers (same as OvO)[21]

**Prediction Efficiency:** Only requires $$K-1$$ classifier evaluations (not all pairwise comparisons)
- Example: 4 classes → only 3 classifier evaluations needed per prediction
- This is much faster than OvO which requires all 6 classifiers

**Path-Dependent:** The prediction path depends on intermediate results

**Advantages :**[21]
- **Efficient prediction:** Only evaluates $$K-1$$ classifiers instead of all $$\frac{K(K-1)}{2}$$
- **Hierarchical structure:** Natural decision-making process
- **No unbalanced data:** Each classifier sees exactly 2 classes (like OvO)
- **Deterministic:** Same input always follows the same path

**Disadvantages:**
- **Order-dependent:** The arrangement of comparisons affects results
- **No error correction:** Early mistakes propagate to final prediction
- **Complex structure:** Requires careful design of the decision tree

**Comparison with OvO:**
- **Training:** Same number of classifiers as OvO
- **Prediction:** Much faster (K-1 vs. K(K-1)/2 evaluations)
- **Accuracy:** Often comparable, but order matters

---

### **4. Unbalanced Decision Tree**

**Definition:** An unbalanced decision tree is a hierarchical multiclass classification structure that is a re-arranged version of One-vs-Rest, providing better handling of imbalanced data.[21]

**Explanation:** Unlike the symmetric DAG structure, an unbalanced decision tree creates an asymmetric hierarchy. It's essentially the One-vs-All strategy organized into a tree structure. At each node, one class is separated from all remaining classes, creating a natural hierarchy that can better handle class imbalance.[21]

---

#### **Structure**

**Example with 4 classes:**

```
                    Root
                     |
              [Class 1 vs Rest]
                /          \
          Class 1      (2,3,4 remain)
                            |
                    [Class 2 vs Rest]
                        /        \
                  Class 2    (3,4 remain)
                                  |
                          [Class 3 vs Rest]
                              /        \
                        Class 3      Class 4
```

**Algorithm:**

1. **Level 1:** Classify "Class 1 vs. All Others (2,3,4)"
   - If Class 1 → Stop, predict Class 1
   - Otherwise → Continue to Level 2

2. **Level 2:** Classify "Class 2 vs. Remaining (3,4)"
   - If Class 2 → Stop, predict Class 2
   - Otherwise → Continue to Level 3

3. **Level 3:** Classify "Class 3 vs. Class 4"
   - Predict the winner

---

#### **Comparison with Standard One-vs-Rest**

| **Aspect** | **Standard OvR** | **Unbalanced Decision Tree** |
|------------|------------------|------------------------------|
| **Structure** | Flat (all classifiers in parallel) | Hierarchical (tree structure) |
| **Prediction** | Evaluate all K classifiers | Early stopping possible |
| **Efficiency** | Always K evaluations | Average fewer evaluations |
| **Imbalance Handling** | Poor (1 vs. K-1 in all nodes) | Better (adaptive hierarchy) |
| **Training** | K binary classifiers | K-1 binary classifiers |

***

#### **Advantages :**[21]

1. **Better Handling of Imbalanced Data:**
   - Can organize tree with rare classes higher up
   - Separate difficult classes early
   - Reduce compound errors

2. **Early Stopping:**
   - If a class is confidently identified at an upper level, no need to evaluate deeper nodes
   - Faster predictions on average

3. **Flexible Structure:**
   - Can arrange based on class frequencies
   - Can group similar classes
   - Adaptable to domain knowledge

4. **Fewer Classifiers Than OvO:**
   - Requires K-1 classifiers instead of $$\frac{K(K-1)}{2}$$

***

#### **Disadvantages:**

1. **Order Sensitivity:**
   - The order of classes in the tree significantly affects performance
   - Poor arrangement can lead to error propagation

2. **Still Some Imbalance:**
   - Lower levels still have 1 vs. multiple classes
   - Not as balanced as OvO

3. **No Voting Mechanism:**
   - Single path to prediction (no error correction from multiple classifiers)

***

#### **Design Strategies:**

**1. Frequency-Based Ordering:**
- Place most frequent classes at top levels
- Reduces average prediction time

**2. Difficulty-Based Ordering:**
- Separate easily distinguishable classes first
- Save hard distinctions for lower levels

**3. Hierarchical Domain Knowledge:**
- Group related classes
- Example: For animal classification:
  ```
  Animals
    ├─ Mammals vs. Non-Mammals
    │   ├─ Cats vs. Dogs
    │   └─ ...
    └─ Birds vs. Reptiles
  ```

***

## **Complete Activation Functions Reference**

### **Summary Comparison Table**

| **Activation** | **Formula** | **Range** | **Best Use Case** | **Advantages** | **Disadvantages** |
|----------------|-------------|-----------|-------------------|----------------|-------------------|
| **Linear** | $$g(z) = z$$ | $$(-\infty, +\infty)$$ | Regression output | Simple, no saturation | No non-linearity, constant gradient |
| **Sigmoid** | $$g(z) = \frac{1}{1+e^{-z}}$$ | $$(0, 1)$$ | Binary classification output | Probabilistic interpretation | Vanishing gradient, not zero-centered |
| **Tanh** | $$g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$ | $$(-1, 1)$$ | Hidden layers (shallow networks) | Zero-centered, stronger gradients | Vanishing gradient for extreme values |
| **ReLU** | $$g(z) = \max(0, z)$$ | $$[0, +\infty)$$ | Hidden layers (deep networks) | Fast, reduces vanishing gradient | Dying ReLU problem, not zero-centered |
| **Leaky ReLU** | $$g(z) = \max(0.01z, z)$$ | $$(-\infty, +\infty)$$ | Hidden layers (alternative to ReLU) | Prevents dying neurons | Requires tuning leak parameter |
| **Softmax** | $$g(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$ | $$(0, 1)$$, sum=1 | Multiclass classification output | Probability distribution | Computationally expensive |

[21]

***

### **Activation Function Decision Guide**

**Choose activation based on your layer and task:**

#### **For Output Layers:**

1. **Binary Classification (2 classes):**
   - Use **Sigmoid**
   - Outputs probability between 0 and 1
   - Apply threshold (typically 0.5) for final decision

2. **Multiclass Classification (K classes):**
   - Use **Softmax**
   - Outputs probability distribution over K classes
   - Choose class with highest probability

3. **Regression (continuous values):**
   - Use **Linear (Identity)**
   - Allows any real-valued output
   - No constraints on range

4. **Multi-Label Classification:**
   - Use **Sigmoid** (independent for each label)
   - Each output is binary decision
   - Can have multiple positive labels

---

#### **For Hidden Layers:**

1. **Deep Neural Networks (modern standard):**
   - Use **ReLU** or **Leaky ReLU**
   - Fast computation
   - Reduces vanishing gradient
   - Most popular choice since 2012

2. **Shallow Neural Networks:**
   - Use **Tanh**
   - Zero-centered helps with gradient flow
   - Better than sigmoid for hidden layers

3. **When ReLU Dies:**
   - Try **Leaky ReLU** or **Parametric ReLU (PReLU)**
   - Small negative slope prevents dead neurons

4. **Specialized Tasks:**
   - **Maxout:** For sparse features
   - **ELU:** For faster learning
   - **Swish/GELU:** For state-of-the-art models

***

### **Practical Implementation Tips**

#### **1. Gradient Descent Convergence**

**Learning Rate Selection:**
- Too small: Slow convergence, many iterations needed
- Too large: Oscillation or divergence
- Typical range: 0.001 to 0.1

**Monitoring Convergence:**
```python
# Check if cost is decreasing
if J_new < J_old:
    continue  # Good progress
else:
    reduce_learning_rate()  # Possible divergence
```

***

#### **2. Feature Scaling**

**Why it matters:**
- Features with different scales cause elongated cost contours
- Gradient descent takes longer to converge
- Some features dominate others

**Normalization Methods:**

**Min-Max Scaling:**
$$
x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

**Standardization (Z-score):**
$$
x_{\text{scaled}} = \frac{x - \mu}{\sigma}
$$

***

#### **3. Regularization Parameter Selection**

**Cross-Validation Approach:**

1. Split data into training and validation sets
2. Try different λ values: [0.001, 0.01, 0.1, 1, 10, 100]
3. Train model with each λ
4. Evaluate on validation set
5. Choose λ with best validation performance

**Visual Inspection:**
- Plot training error vs. λ
- Plot validation error vs. λ
- Choose λ at the "sweet spot" where validation error is minimized

***

### **Complete Code Example: Logistic Regression from Scratch**

```python
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=0):
        """
        Initialize logistic regression model
        
        Parameters:
        - learning_rate: Step size for gradient descent
        - n_iterations: Number of training iterations
        - regularization: Lambda parameter (0 = no regularization)
        """
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = regularization
        self.theta = None
        self.cost_history = []
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X, y, theta):
        """
        Compute regularized logistic regression cost
        
        J(θ) = -1/m * Σ[y*log(h) + (1-y)*log(1-h)] + λ/(2m) * Σθ²
        """
        m = len(y)
        h = self.sigmoid(X @ theta)
        
        # Log loss (cross-entropy)
        epsilon = 1e-5  # Prevent log(0)
        cost = (-1/m) * np.sum(
            y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)
        )
        
        # Add regularization term (exclude theta[0])
        reg_term = (self.lambda_reg / (2*m)) * np.sum(theta[1:]**2)
        
        return cost + reg_term
    
    def compute_gradient(self, X, y, theta):
        """
        Compute gradient for regularized logistic regression
        
        ∂J/∂θⱼ = 1/m * Σ(h - y)*xⱼ + λ/m * θⱼ
        """
        m = len(y)
        h = self.sigmoid(X @ theta)
        
        # Compute gradient
        gradient = (1/m) * (X.T @ (h - y))
        
        # Add regularization term (exclude theta[0])
        gradient[1:] += (self.lambda_reg / m) * theta[1:]
        
        return gradient
    
    def fit(self, X, y):
        """
        Train logistic regression using gradient descent
        
        Parameters:
        - X: Feature matrix (m samples × n features)
        - y: Target vector (m samples)
        """
        # Add intercept term (x0 = 1)
        m, n = X.shape
        X = np.column_stack([np.ones(m), X])
        
        # Initialize parameters
        self.theta = np.zeros(n + 1)
        
        # Gradient descent
        for iteration in range(self.n_iterations):
            # Compute gradient
            gradient = self.compute_gradient(X, y, self.theta)
            
            # Update parameters
            self.theta -= self.lr * gradient
            
            # Track cost
            cost = self.compute_cost(X, y, self.theta)
            self.cost_history.append(cost)
            
            # Print progress
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities for samples"""
        # Add intercept term
        X = np.column_stack([np.ones(len(X)), X])
        return self.sigmoid(X @ self.theta)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels (0 or 1)"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def plot_cost_history(self):
        """Visualize cost function over iterations"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost J(θ)')
        plt.title('Cost Function Convergence')
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000, regularization=0.1)
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nTraining Accuracy: {accuracy:.2%}")
    
    # Plot cost history
    model.plot_cost_history()
```

***

## **Final Summary and Key Takeaways**

### **Core Concepts Mastered**

**1. Logistic Regression Fundamentals:**
- Classification algorithm for categorical outputs
- Uses sigmoid activation to bound predictions between 0 and 1
- Outputs represent probabilities: $$P(y=1|x;\theta)$$

**2. Cost Function:**
- Cannot use MSE (creates non-convex surface)
- Use log loss (cross-entropy): convex and differentiable
- Penalizes confident wrong predictions exponentially

**3. Optimization:**
- Gradient descent iteratively minimizes cost
- Update rule looks identical to linear regression but uses different hypothesis
- Normal equation provides closed-form solution for linear regression

**4. Regularization:**
- Prevents overfitting by penalizing large parameters
- L2 regularization adds $$\frac{\lambda}{2m}\sum\theta_j^2$$ to cost
- Balances fitting data vs. keeping model simple
- Crucial hyperparameter: λ (tune via cross-validation)

**5. Multiclass Classification:**
- **One-vs-All:** K classifiers, each separating one class from rest
- **One-vs-One:** $$\frac{K(K-1)}{2}$$ classifiers, voting mechanism
- **DAG:** Hierarchical elimination, efficient prediction
- **Unbalanced Tree:** Re-arranged OvR with better imbalance handling

**6. Activation Functions:**
- **Sigmoid:** Binary classification output
- **Softmax:** Multiclass classification output
- **ReLU:** Standard for hidden layers in deep networks
- **Tanh:** Hidden layers in shallow networks
- **Linear:** Regression output

---

### **Practical Guidelines**

**When Training Models:**
1. Always normalize/standardize features
2. Start with moderate learning rate (0.01-0.1)
3. Monitor cost function for convergence
4. Use regularization to prevent overfitting
5. Validate with held-out test set

**When Choosing Algorithms:**
- Binary problems → Logistic regression with sigmoid
- Multiclass with many classes → One-vs-All (simplest)
- Multiclass with class imbalance → Unbalanced Decision Tree
- Need maximum accuracy → One-vs-One (but slower)

**When Selecting Hyperparameters:**
- Use k-fold cross-validation
- Grid search over λ values: [0.001, 0.01, 0.1, 1, 10]
- Plot learning curves to diagnose bias vs. variance
- Balance model complexity with generalization

***

### **Reading Materials and Further Study**

**Recommended Resources :**[21]

1. **Textbooks:**
   - Kevin Murphy: "Machine Learning: A Probabilistic Perspective"
   - Chapter 8: Logistic Regression

2. **Online Resources:**
   - Wikipedia: Logistic Regression (comprehensive mathematical treatment)
   - Stanford CS229: Lecture notes on classification

3. **Video Tutorials:**
   - YouTube: "Logistic Regression - Introduction" (Video 7)
   - Link: https://youtu.be/gNhogKJ_q7U

4. **Research Papers:**
   - Feature Selection Methods and Results by Ali Hassan (Chapter 5)

***

<div align="center">

**📚 End of Comprehensive Machine Learning Notes 📚**

### **Topics Covered:**
✅ Logistic Regression Fundamentals  
✅ Convex Cost Functions & Why MSE Fails  
✅ Activation Functions (Linear, ReLU, Sigmoid, Tanh, Softmax)  
✅ Gradient Descent Optimization  
✅ Overfitting & Underfitting  
✅ Regularization (L2)  
✅ Regularized Linear & Logistic Regression  
✅ Normal Equation  
✅ Multiclass Classification (OvA, OvO, DAG, Unbalanced Tree)  


---




[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77906941/1c6cce33-a997-4764-81ab-41d50756f639/Lecture-5-Logistic-Regression-with-Regularizer.pdf)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77906941/8945e5cf-43bd-4424-9d0a-b1e6793f539d/Pre-Lecture-6-Feature-Selection-Dimensionality-Reduction.pdf)
