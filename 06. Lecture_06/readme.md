# Machine Learning: Feature Selection & Dimensionality Reduction

## 📘 Chapter Overview
This chapter covers techniques for selecting the most relevant features from a dataset and reducing its dimensionality while preserving important information. These methods help improve model performance, reduce computational cost, and avoid overfitting.

---

## 🔍 Feature Selection

### Definition
**Feature Selection** is the process of selecting a subset of relevant features from the original set while ignoring or paying less attention to the rest.

### Why Feature Selection is Important
- **Eliminates redundant and irrelevant features**
- **Reduces the curse of dimensionality** - classifier requires fewer training examples
- **Faster and more accurate classification**
- **Improves model interpretability**

### Example
Given a dataset with 28 features for satellite image classification, feature selection helps identify which features contribute most to accurate classification while discarding less important ones.

---

## 🧮 Feature Selection Methods

### Definition
Feature selection is an **optimization problem** where we search through possible feature subsets to find the optimal or near-optimal combination based on specific criteria.

### Search Strategies
- **Exhaustive Search** - Examines all possible subsets
- **Heuristic Search** - Uses rules to guide the search
- **Randomized Search** - Uses random sampling methods

### Evaluation Strategies
- **Filter Methods** - Evaluation independent of classification algorithm
- **Wrapper Methods** - Evaluation uses criterion related to classification algorithm

---

## 🔎 Exhaustive Search

### Definition
**Exhaustive Search** examines all possible feature subsets of size k from n features.

### Mathematical Formulation
- Number of subsets to examine: `C(n,k) = n!/(k!(n-k)!)`
- Selects the subset that performs best according to evaluation function

### Disadvantages
- **Computationally expensive** - grows combinatorially
- **Impractical** for large feature sets
- Cannot guarantee optimal subset with iterative procedures

---

## 📊 Naive Search

### Definition
**Naive Search** sorts features by their individual performance and selects the top k features.

### Process
1. Sort n features by probability of correct recognition
2. Select top k features from sorted list

### Disadvantages
- **Ignores feature correlation**
- Best pair of features may not contain the best individual feature
- **Suboptimal** because it doesn't consider feature interactions

---

## 🧬 Feature Selection using Genetic Algorithms (GAs)

### Definition
**Genetic Algorithms** provide a framework for feature selection using evolutionary principles.

### Process Flow
```
Pre-Processing → Feature Extraction → Feature Selection (GA) → Classifier
```

### Advantages
- Simple, general, and powerful framework
- Can handle large search spaces
- Finds good solutions without exhaustive search

---

## ⚖️ Evaluation Strategies

### Filter Methods
- **Evaluation independent** of classification algorithm
- Uses statistical measures
- **Faster but less accurate**

### Wrapper Methods
- **Evaluation uses criterion** related to classification algorithm
- Uses the actual classifier performance
- **More accurate but computationally expensive**

---

## ➡️ Sequential Forward Selection (SFS)

### Definition
**SFS** is a heuristic search method that starts with an empty set and adds features one by one.

### Algorithm Steps
1. Start with empty feature set
2. Add the best single feature
3. Add features that improve performance when combined with current set
4. Continue until stopping criterion met

### Pseudo Code
```python
S⁰ ← {}
old_criterion ← 0
new_criterion ← 1
while new_criterion > old_criterion:
    old_criterion ← J(Sᵈ)
    f⁺ ← argmax J(Sᵈ + fᵢ) for fᵢ ∉ S
    Sᵈ⁺¹ ← Sᵈ ∪ f⁺
    new_criterion ← J(Sᵈ⁺¹)
    d ← d + 1
```

### Performance
- **Works best when optimal subset is small**
- Cannot remove features once added

---

## ⬅️ Sequential Backward Selection (SBS)

### Definition
**SBS** is a heuristic search method that starts with all features and removes them one by one.

### Algorithm Steps
1. Start with all features
2. Remove the worst feature
3. Continue removing features that hurt performance least
4. Stop when predefined number of features remain

### Performance
- **Works best when optimal subset is large**
- Cannot add features once removed

---

## 🔄 Plus-L Minus-R Selection (LRS)

### Definition
**LRS** generalizes SFS and SBS by adding L features and removing R features in each iteration.

### Two Modes
- **If L > R**: Starts from empty set, adds L features, removes R features
- **If L < R**: Starts from full set, removes R features, adds L features

### Advantages
- Provides **backtracking capabilities**
- Compensates for SFS and SBS weaknesses

### Challenge
- How to choose optimal L and R values?

---

## 🎯 Sequential Floating Selection

### Definition
**Floating Methods** extend LRS with flexible backtracking, dynamically determining how many features to add or remove.

### Types
- **SFFS (Sequential Floating Forward Selection)**
  - Starts from empty set
  - After each forward step, performs backward steps while objective function increases
- **SFBS (Sequential Floating Backward Selection)**
  - Starts from full set
  - After each backward step, performs forward steps while objective function increases

### Advantage
- **Dimensionality "floats" up and down** during search
- More flexible than fixed LRS

---

## ↔️ Bidirectional Search (BDS)

### Definition
**BDS** applies SFS and SBS simultaneously from opposite directions.

### Process
- **SFS** performed from empty set
- **SBS** performed from full set
- Features selected by SFS are not removed by SBS
- Features removed by SBS are not selected by SFS
- Continues until both methods converge to same solution

---

## 📉 Dimensionality Reduction

### Motivation
- Exhaustive search and wrapper methods are **computationally expensive**
- Filter methods are **suboptimal**
- Need automated methods to handle high-dimensional data

### Key Insight
Data often varies in only **limited directions**, allowing compression without significant information loss.

---

## 🗜️ Data Compression

### Definition
**Data Compression** reduces data from higher to lower dimensions while preserving essential information.

### Mathematical Representation
```
x⁽¹⁾ → z⁽¹⁾
x⁽²⁾ → z⁽²⁾
...
x⁽ᵐ⁾ → z⁽ᵐ⁾
```

Where z has lower dimensionality than x.

---

## 📈 PCA is Not Linear Regression

### Key Difference
- **Linear Regression**: Minimizes vertical distance to line
- **PCA**: Minimizes perpendicular distance to line (orthogonal projection)

### Visual Difference
```
Linear Regression:    PCA:
   y ↑                 x₂ ↑
     | ×                 | ×
     |   ×               |   ×
     |     ×             |     ×
     +-------→ x₁        +-------→ x₁
```

---

## 🎯 Principal Component Analysis (PCA)

### Definition
**PCA** is the most common dimensionality reduction method that finds new uncorrelated variables (principal components) that capture maximum variance.

### Properties of Principal Components
- **Linear combinations** of original variables
- **Uncorrelated** with one another
- **Orthogonal** in original space
- **Capture maximum variance** in data

### Process
1. Find direction of greatest variability (PC1)
2. Find next orthogonal direction of greatest variability (PC2)
3. Continue for subsequent components

---

## 📊 Statistical Background

### Mean
**Average value** of dataset
```
μ = (1/m) × Σ Xᵢ
```

### Standard Deviation
**Average distance from mean**
```
σ = √[Σ(Xᵢ - μ)²/(n-1)]
```

### Variance
**Square of standard deviation**
```
σ² = Σ(Xᵢ - μ)²/(n-1)
```

### Covariance
**Measure of how two variables change together**
```
cov(X,Y) = Σ(Xᵢ - μₓ)(Yᵢ - μᵧ)/(n-1)
```

### Properties
- `cov(X,Y) = cov(Y,X)` (symmetric)
- `cov(X,X) = var(X)` (covariance with itself is variance)

---

## 📋 Covariance Matrix

### Definition
Matrix containing all pairwise covariances between variables.

### For 3D data (x,y,z):
```
    [ cov(x,x)  cov(x,y)  cov(x,z) ]
C = [ cov(y,x)  cov(y,y)  cov(y,z) ]
    [ cov(z,x)  cov(z,y)  cov(z,z) ]
```

### Number of Covariance Values
For n dimensions: `n!/((n-2)! × 2)` unique covariance values

---

## 🔢 Eigen Values and Eigen Vectors

### Definition
For a square matrix A, if:
```
A × v = λ × v
```
Then:
- **v** is eigenvector (direction doesn't change)
- **λ** is eigenvalue (scaling factor)

### Properties
- **Only for square matrices**
- **n×n matrix has n eigenvectors**
- Eigenvectors are **orthogonal**
- Usually calculated as **unit vectors** (magnitude = 1)

### Example
```
[2 3] × [3] = [12] = 4 × [3]
[2 1]   [2]   [8]       [2]
```
Eigenvalue = 4, Eigenvector = [3, 2]ᵀ

---

## 🛠️ PCA Step-by-Step Procedure

### Step 1: Get Data
Original dataset with multiple variables.

### Step 2: Subtract Mean
Create zero-mean data by subtracting mean from each variable.

### Step 3: Calculate Covariance Matrix
Compute covariance between all variable pairs.

### Step 4: Calculate Eigenvectors and Eigenvalues
Find principal components from covariance matrix.

### Step 5: Select Principal Components
Choose top k eigenvectors based on eigenvalues.

### Step 6: Transform Data
Project original data onto selected principal components.

---

## 📐 Principal Components, Variance and Least-Squares

### Variance Capture
- **First PC** retains greatest variation
- **k-th PC** retains k-th greatest fraction of variation
- **k-th largest eigenvalue** = variance along k-th PC

### Least-Squares Interpretation
PCs are series of **linear least squares fits**, each orthogonal to previous ones.

---

## 🎯 How to Select Number of Principal Components (k)

### Variance-Based Selection
Keep enough components to capture desired percentage of total variance:
```
(Σᵢ₌₁ᵏ sᵢᵢ) / (Σᵢ₌₁ⁿ sᵢᵢ) ≥ 0.99
```
Where sᵢᵢ are diagonal elements of covariance matrix.

### Scree Plot Method
Plot eigenvalues and look for "elbow" point where curve flattens.

---

## 👤 Facial Recognition using Eigenfaces

### Application
PCA applied to face recognition using **eigenfaces**.

### Process
1. **Face Detection** - Locate faces in images
2. **Face Recognition** - Identify individuals
3. **Face Verification** - Verify claimed identity

### Algorithm
- Convert N×N image to N²×1 vector
- Represent face in low-dimensional space:
```
Φ - mean = w₁u₁ + w₂u₂ + ... + wₖuₖ (where k << N²)
```

### Reconstruction
Face can be reconstructed from eigenfaces:
```
Reconstructed Face = mean + w₁u₁ + w₂u₂ + ... + wₖuₖ
```

### Dimensionality Trick
Instead of computing eigenvectors of huge matrix AAᵀ, compute eigenvectors of smaller matrix AᵀA, then multiply by A.

---

## 💻 Python Implementation Example

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

# Load face images
lfw_people = fetch_lfw_people(min_faces_per_person=4, resize=0.4)
faces = lfw_people.images

# Reshape for PCA
n_samples, h, w = faces.shape
X = faces.reshape(n_samples, -1)

# Perform PCA
n_components = 4
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
pca.fit(X)

# Get eigenfaces
eigenfaces = pca.components_.reshape((n_components, h, w))

# Transform and reconstruct
X_pca = pca.transform(X)
reconstructed_faces = pca.inverse_transform(X_pca)
reconstructed_faces = reconstructed_faces.reshape(n_samples, h, w)
```

---

## 📚 Reading Material

1. **Chapter 5: Feature Selection Methods and Results** by Ali Hassan
2. **Feature Subset Selection in Variable Cost Domains** - Cambridge University Technical Report
3. **A tutorial on Principal Components Analysis** by Lindsay I Smith

---

## 🎯 Key Takeaways

- **Feature Selection** helps identify most relevant features
- **Wrapper methods** are more accurate but computationally expensive
- **Filter methods** are faster but less accurate
- **PCA** reduces dimensionality while preserving variance
- **Eigenfaces** demonstrate practical application of PCA
- Proper feature selection and dimensionality reduction improve model performance and interpretability

This comprehensive guide covers all essential concepts from feature selection to dimensionality reduction, providing both theoretical understanding and practical implementation knowledge.
