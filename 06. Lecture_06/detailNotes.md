---

# **CS-817 Machine Learning: Feature Selection & Dimensionality Reduction**
## **MSc in Artificial Intelligence and Data Sciences**

<div align="center">

https://www.perplexity.ai/search/master-prompt-creating-compreh-KgeaRraiQsWdcDKCIejtyQ#0





</div>

***

## **Table of Contents**
1. [Introduction to Feature Selection](#introduction-to-feature-selection)
2. [Feature Selection Methods](#feature-selection-methods)
3. [Search Strategies](#search-strategies)
4. [Evaluation Strategies](#evaluation-strategies)
5. [Sequential Selection Methods](#sequential-selection-methods)
6. [Advanced Selection Techniques](#advanced-selection-techniques)
7. [Dimensionality Reduction](#dimensionality-reduction)
8. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
9. [Statistical Foundations](#statistical-foundations)
10. [PCA Implementation](#pca-implementation)

***

## **Introduction to Feature Selection**

### **What is Feature Selection?**

**Definition:** Feature selection is the process of identifying and selecting a subset of relevant features (variables, predictors) from a larger set of features that are most useful for building accurate predictive models while eliminating redundant or irrelevant features.[1]

**Explanation:** Imagine you're trying to predict house prices. You might have 100 different features: number of bedrooms, square footage, neighborhood, color of the front door, number of clouds in the sky on the day of sale, etc. Not all these features are equally important. Feature selection helps you pick only the most meaningful ones (like bedrooms and square footage) while ignoring irrelevant ones (like cloud count). This is essentially asking: *"What makes a good set of features for classification or prediction?"*[1]

The core goal is to select a subset of size **k** features from **n** total features (where k < n) that leads to the smallest classification error.[1]

**Example:**
- If you have 50 features describing medical patients, feature selection might identify that only 10 features (age, blood pressure, cholesterol, etc.) are truly predictive of heart disease, while the other 40 add noise.

***

### **The Curse of Dimensionality**

**Definition:** The curse of dimensionality refers to various phenomena that arise when analyzing data in high-dimensional spaces that do not occur in low-dimensional settings.[1]

**Explanation:** As the number of features (dimensions) increases, the amount of data needed to make reliable predictions grows exponentially. Think of it like this: in 1D (a line), you need only a few points to understand the pattern. In 2D (a plane), you need more points. In 3D (space), even more. When you have 100 dimensions, you need astronomical amounts of data! The data becomes so sparse in high-dimensional space that machine learning models struggle to find meaningful patterns.[1]

**Example:**
- With 2 features and 10 samples per feature value, you need 10² = 100 training examples.
- With 10 features, you'd need 10¹⁰ = 10 billion examples!
- This makes training impractical and expensive.

***

### **Why is Feature Selection Important?**

Feature selection provides three critical benefits :[1]

1. **Eliminates Redundant and Irrelevant Features:**
   - **Definition:** Redundant features are those that provide the same information as other features. Irrelevant features have no predictive power.
   - **Explanation:** If you have "height in centimeters" and "height in inches" as two features, they're redundant—they say the same thing. If you're predicting house prices and include "owner's favorite color," that's likely irrelevant.
   - **Example:** In email spam detection, the words "buy" and "purchase" might be redundant features. The word "the" is likely irrelevant.

2. **Requires Fewer Training Examples:**
   - **Explanation:** By reducing dimensions, you need less data to train accurate models, directly addressing the curse of dimensionality.
   - **Example:** Instead of needing 10,000 examples with 100 features, you might only need 1,000 examples with 10 carefully selected features.

3. **Enables Faster and More Accurate Classification:**
   - **Explanation:** Fewer features mean less computation time during both training and prediction. Additionally, removing noisy or irrelevant features often improves model accuracy.
   - **Example:** A medical diagnosis system that evaluates 10 key symptoms instead of 100 measurements will make predictions faster and potentially more accurately.

***

## **Feature Selection Methods**

### **Feature Selection as an Optimization Problem**

**Definition:** Feature selection is framed as an optimization problem where we search through the space of all possible feature subsets to find the one that optimizes a specific criterion (like classification accuracy).[1]

**Explanation:** Think of feature selection like searching for the best team of players. You have many players (features), but you can only pick a certain number for your starting lineup (subset). You need a strategy to search through all possible team combinations and a way to evaluate which team performs best. Feature selection works the same way with two key components :[1]

1. **Search Strategies:** How do we explore the space of possible feature combinations?
2. **Evaluation Strategies:** How do we measure which combination is best?

***

## **Search Strategies**

Search strategies determine how we navigate through the vast space of possible feature combinations.[1]

***

### **1. Exhaustive Search**

**Definition:** Exhaustive search examines every possible combination of features to find the absolute best subset.[1]

**Explanation:** This is the "try everything" approach. If you have **m** features and want to select **k** features, exhaustive search evaluates all $$\binom{m}{k}$$ (read as "m choose k") possible combinations. While this guarantees finding the optimal subset, it's computationally impractical for real-world problems because the number of combinations grows combinatorially.[1]

**Formula:**
$$
\text{Number of subsets} = \binom{m}{k} = \frac{m!}{k!(m-k)!}
$$

**Example:**
- With 10 features, selecting 5 requires evaluating $$\binom{10}{5} = 252$$ combinations—manageable.
- With 50 features, selecting 25 requires evaluating $$\binom{50}{25} \approx 1.26 \times 10^{14}$$ combinations—impossible!

**Limitations:**
- Computationally infeasible for large feature sets
- Time complexity grows exponentially
- Iterative procedures are used as alternatives, though they don't guarantee optimal solutions[1]

***

### **2. Naive Search**

**Definition:** Naive search ranks all features individually by their performance and selects the top k features.[1]

**Explanation:** This is the simplest approach. You evaluate each feature independently (e.g., by how well each one alone predicts the target) and pick the top performers. It's fast but has a critical flaw: it ignores how features work together. The best pair of features might not include the single best feature![1]

**Algorithm:**
1. Calculate the probability of correct recognition for each feature individually
2. Sort features in descending order of performance
3. Select the top k features from this sorted list

**Example:**
- In predicting student performance, "hours studied" might be the best single feature.
- "Hours studied" AND "previous test scores" together might be the best pair.
- But "attendance rate" AND "previous test scores" might actually be the best pair overall, even though attendance alone isn't the top feature.

**Disadvantage:**
- **Feature correlation is not considered:** The method treats features as independent when they often work together.[1]
- **The best pair may not contain the best individual feature:** This is a critical oversight in many real-world scenarios.[1]

***

### **3. Random Search Using Genetic Algorithms (GAs)**

**Definition:** Genetic Algorithms are randomized search techniques inspired by biological evolution that explore the feature space by mimicking natural selection, crossover, and mutation.[1]

**Explanation:** GAs treat feature selection like evolution in nature. Each possible subset of features is like an organism with a "genome" (binary string where 1 = feature included, 0 = feature excluded). The algorithm:

1. Creates a population of random feature subsets
2. Evaluates each subset's "fitness" (performance)
3. "Breeds" the best subsets (crossover)
4. Randomly mutates some subsets (to explore new possibilities)
5. Repeats until convergence

GAs provide a **simple, general, and powerful framework** for feature selection without exhaustively searching every possibility.[1]

**Example:**
- **Initial Population:** 
  - Subset A:  (features 1, 3, 5 selected)[1]
  - Subset B:  (features 2, 3, 4 selected)[1]
- **Crossover:** Combine A and B to create:  (features 1, 3, 4)[1]
- **Mutation:** Randomly flip one bit:  (features 1, 4)[1]

**Advantages:**
- Explores feature space efficiently
- Can escape local optima through mutation
- Doesn't require gradient information
- Applicable to any evaluation criterion

***

## **Evaluation Strategies**

Evaluation strategies determine how we measure the quality of a selected feature subset.[1]

***

### **1. Filter Methods**

**Definition:** Filter methods evaluate features independent of any specific classification algorithm, using statistical measures or information-theoretic criteria.[1]

**Explanation:** Filter methods are like screening candidates before an interview. You use general criteria (education, experience) without testing them in the actual job. Similarly, filter methods use statistical measures like correlation, mutual information, or chi-square tests to score features before any model is trained. They're fast but may miss feature combinations that work well specifically with your chosen classifier.[1]

**Characteristics:**
- **Algorithm-independent:** Evaluation doesn't depend on the classifier
- **Fast:** No need to train models repeatedly
- **General:** Same features work for different algorithms

**Example:**
- Use correlation coefficients to filter out features with low correlation to the target variable
- Use chi-square tests to identify features with statistical independence from the target

***

### **2. Wrapper Methods**

**Definition:** Wrapper methods evaluate feature subsets by actually training and testing a specific classification algorithm, using the classifier's performance as the evaluation criterion.[1]

**Explanation:** Wrapper methods are like actually testing candidates on the job. You try different feature subsets by training your classifier (SVM, neural network, etc.) on each subset and measuring its accuracy. This is more accurate than filter methods because you're testing what actually matters—classifier performance—but it's computationally expensive because you train many models.[1]

**Characteristics:**
- **Algorithm-dependent:** Evaluation is tailored to your specific classifier
- **More accurate:** Directly measures what you care about (classifier performance)
- **Computationally expensive:** Must train and test classifier for each subset[1]

**Example:**
- For each feature subset, train a Random Forest classifier
- Evaluate using 5-fold cross-validation accuracy
- Select the subset with highest accuracy

**Comparison:**
Wrapper methods provide **more accurate solutions** than filter methods but are generally **more computationally expensive**.[1]

***

## **Sequential Selection Methods**

Sequential methods are **heuristic search strategies** that build or reduce feature subsets iteratively, one feature at a time.[1]

***

### **1. Sequential Forward Selection (SFS)**

**Definition:** SFS is a bottom-up greedy search algorithm that starts with an empty set and iteratively adds the best feature at each step.[1]

**Explanation:** SFS is like building a team by adding one player at a time, always choosing the player who most improves team performance. You start with zero features, try adding each remaining feature one by one, and permanently add the one that gives the best performance. Then repeat with the remaining features until you reach a stopping criterion.[1]

**Algorithm Steps :**[1]
1. **Step 1:** Start with an empty feature set
2. **Step 2:** Evaluate all single features and select the best one
3. **Step 3:** Form pairs by adding one remaining feature to the best single feature; select the best pair
4. **Step 4:** Form triplets by adding one remaining feature to the best pair; select the best triplet
5. **Continue:** Repeat until the evaluation criterion stops improving or k features are selected

**Key Characteristics:**
- **Greedy approach:** Makes locally optimal choices at each step
- **Bottom-up:** Builds from empty set
- **Best for small optimal subsets:** SFS performs best when the optimal subset is small[1]

**Example:**
In satellite image classification with 28 features :[1]
- **Iteration 1:** Feature 15 selected (70% accuracy)
- **Iteration 2:** Features 15, 7 selected (75% accuracy)
- **Iteration 3:** Features 15, 7, 23 selected (78% accuracy)
- **Iteration 10:** Best accuracy of 82% achieved
- The process shows classification accuracy on x-axis and features added on y-axis

**Limitations:**
- **No backtracking:** Once a feature is added, it cannot be removed (even if it becomes redundant later)
- **Nesting problem:** Can get stuck with suboptimal early choices
- May not find the global optimum

***

### **2. Sequential Backward Selection (SBS)**

**Definition:** SBS is a top-down greedy search algorithm that starts with all features and iteratively removes the worst feature at each step.[1]

**Explanation:** SBS is the opposite of SFS—it's like starting with a full team and removing the weakest player at each step. You begin with all **d** features, try removing each feature one by one, and permanently remove the one whose absence improves (or least hurts) performance. Repeat until you reach the desired number of features.[1]

**Algorithm Steps :**[1]
1. **Step 1:** Start with all d features and compute the criterion function
2. **Step 2:** Remove each feature one at a time, evaluate all (d-1) feature subsets, and discard the worst feature
3. **Step 3:** From remaining (d-1) features, remove each one at a time, and discard the worst to form (d-2) features
4. **Continue:** Repeat until a predefined number of features remain

**Key Characteristics:**
- **Greedy approach:** Makes locally optimal choices at each step
- **Top-down:** Reduces from full set
- **Best for large optimal subsets:** SBS performs best when the optimal subset is large[1]

**Example:**
In satellite image classification with 28 features :[1]
- **Iteration 1:** Remove feature 22 (74% accuracy)
- **Iteration 2:** Remove features 22, 11 (76% accuracy)
- **Iteration 15:** Best accuracy of 83% achieved
- The process shows classification accuracy on x-axis and features removed on y-axis

**Limitations:**
- **No backtracking:** Once a feature is removed, it cannot be added back
- **Nesting problem:** Mirror of SFS—can get stuck with suboptimal late choices
- Computationally expensive for large feature sets

***

## **Advanced Selection Techniques**

***

### **1. Plus-L Minus-R Selection (LRS)**

**Definition:** LRS is a generalization of SFS and SBS that adds L features and removes R features in each iteration, providing limited backtracking capability.[1]

**Explanation:** LRS attempts to fix the "no backtracking" problem of SFS and SBS. Instead of just adding one feature (SFS) or removing one feature (SBS), LRS adds multiple features and removes multiple features in each step. This gives it the ability to reconsider earlier decisions to some extent.[1]

**Algorithm :**[1]
- **If L > R:** 
  - Start from empty set
  - Repeatedly add L features, then remove R features
  - Net effect: adding (L-R) features per iteration
  
- **If L < R:**
  - Start from full set
  - Repeatedly remove R features, then add L features
  - Net effect: removing (R-L) features per iteration

**Example:**
- **L=3, R=1 (forward):** Add 3 best features, remove 1 worst → net gain of 2 features
- **L=1, R=2 (backward):** Remove 2 worst features, add 1 best → net loss of 1 feature

**Benefits:**
- Provides **backtracking capabilities** to compensate for weaknesses of pure SFS/SBS[1]
- Can correct suboptimal earlier decisions

**Challenge:**
- **How to choose optimal values of L and R?**[1]
- No clear guidelines; often requires trial and error or domain expertise

***

### **2. Sequential Floating Selection (SFFS and SFBS)**

**Definition:** Sequential floating selection methods are extensions of LRS with **flexible, data-driven backtracking** where the values of L and R are determined dynamically from the data rather than fixed in advance.[1]

**Explanation:** The "floating" refers to how the dimensionality of the feature subset can float up and down during the search. Unlike LRS where L and R are fixed, floating methods intelligently decide at each step whether to add or remove features based on what improves performance. This provides more sophisticated backtracking than LRS.[1]

**Two Variants:**

#### **Sequential Floating Forward Selection (SFFS)**

**Algorithm :**[1]
1. Start from the empty set
2. Perform one forward step (add best feature)
3. Perform backward steps as long as the objective function increases
4. Repeat until stopping criterion met

**Explanation:** SFFS primarily moves forward (adds features) but checks after each addition whether removing any previously added feature would improve performance. If so, it removes features until no more improvements are possible, then continues adding.[1]

#### **Sequential Floating Backward Selection (SFBS)**

**Algorithm :**[1]
1. Start from the full set
2. Perform one backward step (remove worst feature)
3. Perform forward steps as long as the objective function increases
4. Repeat until stopping criterion met

**Explanation:** SFBS primarily moves backward (removes features) but checks after each removal whether adding back any previously removed feature would improve performance.[1]

**Advantages:**
- **Flexible backtracking:** Adapts to data structure
- **Better than LRS:** More sophisticated than fixed L and R values
- **Escapes local optima:** Can undo poor earlier decisions more effectively

**Example:**
- SFFS might add features 1, 3, 5, then realize removing feature 1 improves performance, then add feature 7, and so on
- The number of features can fluctuate (float) during the search

***

### **3. Bidirectional Search (BDS)**

**Definition:** BDS applies SFS and SBS simultaneously from opposite ends, with constraints ensuring they converge to the same solution.[1]

**Explanation:** Imagine two search teams starting from opposite ends of a tunnel, working toward each other. BDS runs SFS from the empty set (adding features) and SBS from the full set (removing features) at the same time. To ensure they don't conflict and eventually meet :[1]

**Algorithm Constraints :**[1]
1. **SFS is performed from the empty set** (bottom-up)
2. **SBS is performed from the full set** (top-down)
3. **Features already selected by SFS cannot be removed by SBS** (protects forward progress)
4. **Features already removed by SBS cannot be selected by SFS** (protects backward progress)

**Benefits:**
- Explores feature space from both directions
- Can converge faster than unidirectional methods
- Leverages strengths of both SFS (good for small subsets) and SBS (good for large subsets)

**Example:**
- **SFS side:** Selects features {1, 5, 7}
- **SBS side:** Removes features {2, 4, 8, 9}
- **Valid final space:** Features {1, 3, 5, 6, 7, 10}
- Both searches converge to optimal subset in this space

***

## **Dimensionality Reduction**

### **Introduction**

**Definition:** Dimensionality reduction is the process of reducing the number of features (dimensions) in a dataset while preserving as much information as possible.[1]

**Explanation:** Unlike feature selection (which picks a subset of existing features), dimensionality reduction creates new features that are combinations of original features. It's motivated by the practical challenges of feature selection :[1]

**Motivation :**[1]
1. **Exhaustive search is very expensive** — computationally non-feasible
2. **Wrapper-based methods (SFS, SBS, SFFS, etc.) are very expensive** — require training many models
3. **Filter-based methods are suboptimal** — don't consider classifier performance
4. **Solution:** Try to automate dimensionality reduction in a data-driven way

***

### **Key Insight: Spread of Data**

**Observation:** Data often varies in only some limited directions.[1]

**Explanation:** When you have high-dimensional data (many features), it's usually the case that most of the meaningful variation happens along just a few directions. Think of a flat sheet of paper in 3D space—the data varies in two dimensions (length and width) but not in the third (thickness). You can't spot this low-dimensional structure by just looking at numbers; you need mathematical tools like PCA.[1]

**Example:**
- Student performance data with 20 features might actually vary along just 3 main directions: "overall ability," "test-taking skills," and "attendance patterns"
- The other 17 features are just noisy variations of these three main patterns

***

### **Data Compression**

**Definition:** Data compression in dimensionality reduction means projecting data onto a lower-dimensional subspace with maximum variation, effectively dropping unnecessary axes and rotating remaining axes.[1]

**Explanation:** Imagine your data as points in a 3D space, but all points lie roughly on a tilted 2D plane. Data compression:
1. **Rotates** the axes so one axis aligns with the plane
2. **Drops** the axis perpendicular to the plane (it has almost no variation)
3. **Keeps** the two axes within the plane

This reduces from 3D to 2D while keeping almost all information.[1]

**Visual Understanding:**
- Original data might have high variance in one direction and low variance in another
- We rotate coordinates to align with directions of maximum variance
- We drop directions with minimal variance

***

### **PCA is NOT Linear Regression**

**Critical Distinction:** Principal Component Analysis (PCA) and Linear Regression are fundamentally different.[1]

**Linear Regression:**
- Predicts a target variable (y) from features (x)
- Minimizes vertical distance (error in y-direction) between points and line
- Has a dependent variable and independent variables
- Goal: prediction or inference about relationships

**PCA:**
- No target variable—unsupervised learning
- Finds directions of maximum variance in data
- Minimizes perpendicular distance from points to principal components
- All variables are treated equally
- Goal: dimensionality reduction or data compression[1]

**Example:**
- **Regression:** Predict house prices (y) from square footage (x)—line minimizes vertical distances to points
- **PCA:** Find main variation in house data (price, size, age)—line captures direction of maximum spread, minimizing perpendicular distances

***

## **Principal Component Analysis (PCA)**

### **Definition and Overview**

**Definition:** PCA is the most common form of dimensionality reduction that transforms original features into a new set of uncorrelated variables (principal components) that capture as much variance as possible.[1]

**Explanation:** PCA is like finding the "best angles" to view your data. Imagine photographing a 3D object—some angles show more detail than others. PCA finds the mathematical "angles" (axes) that capture the most information about your data. These new axes are combinations of your original features.[1]

***

### **Key Properties of Principal Components**

The new variables/dimensions created by PCA have five important properties :[1]

1. **Linear combinations of original features:**
   - Each principal component is a weighted sum of original features
   - Example: PC1 = 0.7×height + 0.5×weight - 0.3×age

2. **Uncorrelated with one another:**
   - Principal components are statistically independent
   - No redundant information between components

3. **Orthogonal in original dimension space:**
   - Principal components are perpendicular to each other
   - This ensures uncorrelated nature geometrically[1]

4. **Capture maximum variance:**
   - First PC captures the most variance
   - Second PC captures the second-most variance (among directions orthogonal to first)
   - And so on...[1]

5. **Called Principal Components (PCs):**
   - The transformed axes are the principal components

***

### **Understanding Principal Components**

#### **First Principal Component (PC1)**

**Definition:** The first principal component is the direction of **greatest variability (covariance)** in the data.[1]

**Explanation:** PC1 is the line through your data that, if you project all points onto it, gives the maximum spread. It's like finding the axis where data is most "stretched out." Projections along PC1 discriminate the data most along any single axis.[1]

**Example:**
- In data about students (study hours vs. test scores), PC1 might be the diagonal direction that captures "overall academic performance"—students vary most along this combined dimension

#### **Second Principal Component (PC2)**

**Definition:** The second principal component is the next orthogonal (perpendicular) direction of greatest variability after removing all variability along PC1.[1]

**Explanation:** After accounting for the main pattern (PC1), PC2 finds the next biggest pattern in the remaining variation. It must be perpendicular to PC1 to ensure independence.[1]

**Example:**
- In student data, PC2 might capture "test-taking skills vs. homework effort"—a pattern independent of overall performance

#### **Subsequent Components**

**Pattern:** Each subsequent principal component follows the same rule—find the direction of maximum remaining variance that's orthogonal to all previous components.[1]

***

### **Visualizing PCA**

**Orthogonal Axes of Maximum Variance:**
The principal components are orthogonal (perpendicular) axes that capture the maximum variance of the data.[1]

**Geometric Interpretation:**
- Imagine a cloud of data points in 2D
- PC1 is the line that best fits through the center of the cloud (most stretch)
- PC2 is perpendicular to PC1, capturing the next most variation
- Together, they form a rotated coordinate system aligned with data's natural shape

***

## **Statistical Foundations**

To understand PCA mathematically, we need several statistical concepts.[1]

***

### **1. Mean (μ)**

**Definition:** The mean is the average value of a dataset.[1]

**Formula:**
$$
\mu_i = \frac{1}{m} \sum_{i=1}^{m} X_i
$$

**Explanation:** The mean tells you the center point of your data. It's calculated by summing all values and dividing by the count.[1]

**Example:**
- Marks: 
- Mean: $$\frac{0 + 8 + 12 + 20}{4} = \frac{40}{4} = 10$$

**Question:** What's the difference between  and ?[1]
**Answer:** Both have the same mean (10), but different spreads. The first varies more.

***

### **2. Standard Deviation (σ) and Variance (σ²)**

**Definition:** 
- **Variance** measures the average squared distance from the mean[1]
- **Standard deviation** is the square root of variance, representing average distance from the mean[1]

**Formulas:**
$$
\sigma^2 = \frac{\sum_{i=1}^{n} (X_i - \mu)^2}{n - 1}
$$

$$
\sigma = \sqrt{\frac{\sum_{i=1}^{n} (X_i - \mu)^2}{n - 1}}
$$

**Explanation:** Variance and standard deviation quantify how "spread out" data is. High values mean data is scattered far from the mean; low values mean it's clustered tightly. We divide by (n-1) for sample variance (Bessel's correction).[1]

**Example:**
- **Dataset 1:**  → High variance (spread out)
- **Dataset 2:**  → Low variance (clustered)

***

### **3. Covariance**

**Definition:** Covariance measures how two variables change together.[1]

**Formula:**
$$
\text{cov}(X, Y) = \frac{\sum_{i=1}^{n} (X_i - \mu_X)(Y_i - \mu_Y)}{n - 1}
$$

**Explanation:** While variance measures spread of one variable, covariance measures the relationship between two variables. It answers: "When X increases, does Y tend to increase (positive covariance), decrease (negative covariance), or is there no pattern (zero covariance)?"[1]

**In English:** For each data point, multiply the difference between X and its mean by the difference between Y and its mean. Add these up and divide by (n-1).[1]

**Key Properties :**[1]
- **Covariance with itself equals variance:** cov(X, X) = var(X)
- **Covariance is commutative:** cov(X, Y) = cov(Y, X)
  - Because (X_i - μ_X)(Y_i - μ_Y) = (Y_i - μ_Y)(X_i - μ_X)—multiplication is commutative[1]

**Example:**
- **Height and Weight:** Usually positive covariance (taller people tend to weigh more)
- **Study Time and Test Scores:** Usually positive covariance
- **Ice Cream Sales and Heating Bills:** Negative covariance (inverse relationship)

***

### **4. Covariance Matrix**

**Definition:** For datasets with more than 2 dimensions, the covariance matrix stores all pairwise covariances.[1]

**Formula for 3D dataset (x, y, z):**
$$
C = \begin{bmatrix}
\text{cov}(x, x) & \text{cov}(x, y) & \text{cov}(x, z) \\
\text{cov}(y, x) & \text{cov}(y, y) & \text{cov}(y, z) \\
\text{cov}(z, x) & \text{cov}(z, y) & \text{cov}(z, z)
\end{bmatrix}
$$

**Explanation:** The covariance matrix is a square matrix where:
- **Diagonal elements** are variances (covariance of each variable with itself)
- **Off-diagonal elements** are covariances between different variables
- The matrix is **symmetric** because cov(x, y) = cov(y, x)[1]

**Number of Unique Covariances:**
For **n** dimensions, you calculate:
$$
\frac{n!}{(n-2)! \times 2}
$$

**Example for n=3:**
$$
\frac{3!}{(3-2)! \times 2} = \frac{6}{1 \times 2} = 3 \text{ unique covariances}
$$
These are: cov(x,y), cov(x,z), cov(y,z)[1]

***

### **5. Eigenvalues and Eigenvectors**

**Definition:** 
- **Eigenvector:** A special vector that only gets scaled (not rotated) when multiplied by a matrix[1]
- **Eigenvalue:** The scaling factor by which the eigenvector is multiplied[1]

**Mathematical Relationship:**
$$
A \mathbf{v} = \lambda \mathbf{v}
$$

Where:
- A = square matrix
- **v** = eigenvector
- λ = eigenvalue

**Explanation:** Normally, when you multiply a vector by a matrix, the vector gets rotated AND scaled. But eigenvectors are special—they only get scaled (stretched or shrunk) without changing direction. The amount of scaling is the eigenvalue.[1]

**Simple Example:**
If multiplying a vector by a matrix scales it by 4 without rotation:
- The vector is an eigenvector
- The eigenvalue is 4[1]

**Key Properties :**[1]

1. **Only for square matrices:** Eigenvectors can only be found for square matrices
2. **n eigenvectors for n×n matrix:** A matrix of size n×n has n eigenvectors
3. **Scaling invariance:** If **v** is an eigenvector, so is any multiple c**v** (direction unchanged)
4. **Orthogonality:** All eigenvectors of a matrix are mutually orthogonal (perpendicular)
5. **Unit eigenvectors:** Typically normalized to unit length (magnitude = 1)

**Normalization Example :**[1]
Original eigenvector:
$$
\begin{bmatrix} 3 \\ 2 \end{bmatrix}
$$

Magnitude:
$$
\sqrt{3^2 + 2^2} = \sqrt{13}
$$

Normalized:
$$
\begin{bmatrix} 3/\sqrt{13} \\ 2/\sqrt{13} \end{bmatrix}
$$

**Geometric Interpretation:**
- Eigenvectors of the covariance matrix point in the directions of maximum variance
- Eigenvalues represent the amount of variance along each eigenvector direction
- In PCA, eigenvectors become the principal components!

***

## **PCA Implementation**

### **Step-by-Step PCA Algorithm**

PCA involves a systematic process to transform data into principal components.[1]

***

#### **Step 1: Get the Data**

**Task:** Organize your dataset into a matrix format.[1]

**Example Data :**[1]
```
x      y
2.5    2.4
0.5    0.7
2.2    2.9
1.9    2.2
3.1    3.0
2.3    2.7
2.0    1.6
1.0    1.1
1.5    1.6
1.1    0.9
```

**Explanation:** Start with your raw data where each row is an observation (data point) and each column is a feature (dimension).

***

#### **Step 2: Subtract the Mean (Zero-Centering)**

**Task:** Subtract the mean of each feature from all data points.[1]

**Why:** PCA requires zero-centered data to correctly compute variance directions. Centering translates the data so its mean is at the origin (0,0).[1]

**Formula for each feature:**
$$
X_{\text{centered}} = X - \mu
$$

**Example Result :**[1]
```
x         y
0.69      0.49
-1.31    -1.21
0.39      0.99
0.09      0.29
1.29      1.09
0.49      0.79
0.19     -0.31
-0.81    -0.81
-0.31    -0.31
-0.71    -1.01
```

**Interpretation:** Now the data is centered around (0,0), ready for variance analysis.

***

#### **Step 3: Calculate the Covariance Matrix**

**Task:** Compute the covariance matrix of the zero-centered data.[1]

**Formula:**
$$
C = \frac{1}{n-1} X^T X
$$

Where X is the zero-centered data matrix.

**Example Result :**[1]
```
C = [0.616555556   0.615444444]
    [0.615444444   0.716555556]
```

**Interpretation:** 
- **Diagonal elements** (0.617, 0.717): Variances of x and y
- **Off-diagonal elements** (0.615): Positive covariance—x and y increase together[1]
- Since non-diagonal elements are positive, both variables increase together[1]

***

#### **Step 4: Calculate Eigenvectors and Eigenvalues**

**Task:** Compute eigenvectors and eigenvalues of the covariance matrix.[1]

**Example Result :**[1]
```
Eigenvalues:
λ₁ = 1.28402771  (larger - more important)
λ₂ = 0.0490833989  (smaller - less important)

Eigenvectors:
v₁ = [-0.677873399]    v₂ = [-0.735178656]
     [-0.735178656]         [0.677873399]
```

**Sorting:** Always sort eigenvectors by decreasing eigenvalue—the eigenvector with the largest eigenvalue is PC1.[1]

**Sign Invariance:** The sign of eigenvectors doesn't matter—flipping the sign gives a vector in the opposite direction but represents the same line.[1]

**Geometric Interpretation :**[1]
- **v₁ (PC1):** Points through the middle of the data cloud (like a line of best fit)
- **v₂ (PC2):** Perpendicular to PC1, captures secondary variation
- Together they form a rotated coordinate system aligned with data's natural structure

**Key Observation :**[1]
- Eigenvectors are **perpendicular** to each other
- One eigenvector goes through the middle of the points (main pattern)
- The second eigenvector shows secondary variation off to the side

***

#### **Step 5: Choose Principal Components**

**Task:** Decide how many principal components to keep.[1]

**Options:**

1. **Keep All Components (No Dimensionality Reduction):**
   - Use all eigenvectors
   - Retains 100% of information
   - Just rotates axes without compression

2. **Keep k Components (Dimensionality Reduction):**
   - Choose only the first k eigenvectors (those with largest eigenvalues)
   - Reduces from n to k dimensions
   - Lose some information, but if eigenvalues are small, you don't lose much[1]

**Feature Vector Formation :**[1]

**Definition:** The feature vector is a matrix formed by stacking chosen eigenvectors as columns.

**Full Feature Vector (keep both PCs):**
$$
\text{FeatureVector} = \begin{bmatrix}
-0.677873399 & -0.735178656 \\
-0.735178656 & 0.677873399
\end{bmatrix}
$$

**Reduced Feature Vector (keep only PC1):**
$$
\text{FeatureVector} = \begin{bmatrix}
-0.677873399 \\
-0.735178656
\end{bmatrix}
$$

**Notation:**
- **Ureduce** = U(:, 1:k) — Select first k columns of eigenvector matrix[1]

***

#### **Step 6: Transform the Data**

**Task:** Project the original data onto the principal components.[1]

**Formula:**
$$
\text{FinalData} = \text{FeatureVector}^T \times \text{ZeroMeanData}^T
$$

Or in matrix notation:
$$
Z = U_{\text{reduce}}^T \times X
$$

Where:
- **FeatureVector^T:** Transposed feature vector (eigenvectors in rows)
- **ZeroMeanData^T:** Transposed zero-centered data
- **Z:** Projected data (dimensionality-reduced)[1]

**Example Result (keeping both PCs) :**[1]
```
        PC1           PC2
    -0.827970186  -0.175115307
     1.77758033    0.142857227
    -0.992197494   0.384374989
    -0.274210416   0.130417207
    -1.67580142   -0.209498461
    -0.912949103   0.175282444
     0.0991094375 -0.349824698
     1.14457216    0.0464172582
     0.438046137   0.0177646297
     1.22382056   -0.162675287
```

**Interpretation:** 
- Each data point now has coordinates in the PC1-PC2 space
- PC1 values show projection on first principal component
- PC2 values show projection on second principal component
- Data is rotated to align with maximum variance directions

***

### **Reconstruction of Original Data**

**Task:** Transform the dimensionality-reduced data back to the original feature space.[1]

**Formula :**[1]
$$
Z = U^T \times X
$$
$$
X' = (U \times Z) + \mu_{\text{original}}
$$

Where:
- **Z:** Projected data in PC space
- **U:** Eigenvector matrix (or U_reduce if dimensionality was reduced)
- **X':** Reconstructed data
- **μ_original:** Original mean (added back to de-center)

**Explanation:** To go back from PC coordinates to original coordinates:
1. Multiply the projected data by the eigenvector matrix
2. Add back the original mean (we subtracted it in Step 2)

**If dimensionality was reduced:** The reconstructed data will be an approximation—you lose the dimensions you discarded.[1]

**Example (using only PC1) :**[1]
```
Reconstructed x values:
-0.827970186
 1.77758033
-0.992197494
-0.274210416
-1.67580142
-0.912949103
 0.0991094375
 1.14457216
 0.438046137
 1.22382056
```

**Note:** Since we kept only PC1, the reconstruction is approximate—the y-dimension information from PC2 is lost.

***

### **Choosing the Number of Principal Components (k)**

**Critical Decision:** How many principal components should we keep?[1]

***

#### **Method 1: Variance Explained**

**Definition:** Choose k such that the retained PCs explain a certain percentage (e.g., 95%, 99%) of total variance.[1]

**Formula :**[1]
$$
\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{n} \lambda_i} \geq 0.99
$$

Or equivalently (using singular values from SVD):
$$
\frac{\sum_{i=1}^{k} s_{ii}}{\sum_{i=1}^{n} s_{ii}} \geq 0.99
$$

Where:
- **λ_i:** Eigenvalues
- **s_ii:** Singular values from Singular Value Decomposition (SVD)
- **k:** Number of PCs to keep
- **n:** Total number of features

**Explanation:** 
- The **kth largest eigenvalue** represents the variance in the sample along the kth PC[1]
- Sum of eigenvalues = total variance in data
- Choose k such that retained PCs capture 95% or 99% of variance

**Example:**
```
Eigenvalues: [1.28, 0.82, 0.65, 0.31, 0.09, 0.05]
Total variance: 3.20
Cumulative variance:
  k=1: 1.28/3.20 = 40%
  k=2: 2.10/3.20 = 66%
  k=3: 2.75/3.20 = 86%
  k=4: 3.06/3.20 = 96% ← Choose k=4
```

**Common Thresholds:**
- **99%:** Very conservative—minimal information loss
- **95%:** Common choice—good balance
- **90%:** More aggressive compression

***

#### **Method 2: Scree Plot**

**Definition:** Plot eigenvalues in decreasing order and look for the "elbow" where the curve flattens.[1]

**Interpretation:**
- Sharp drop: Important components
- Flat region: Less important components
- Keep components before the elbow

***

#### **Method 3: Application-Specific**

**Approach:** Test different values of k and evaluate performance on your specific task.[1]

**Example:**
- Train classifier with k=5, 10, 15, 20 components
- Choose k with best validation accuracy

***

## **PCA Applications**

### **Facial Recognition Using Eigenfaces**

PCA has a famous application in facial recognition called "Eigenfaces".[1]

***

#### **Types of Face Tasks**

1. **Face Detection:** Is there a face in the image?[1]
2. **Face Recognition:** Who is this person?[1]
3. **Face Verification:** Is this person who they claim to be?[1]
4. **Facial Identification:** Match face to identity in database[1]

***

#### **Eigenfaces Concept**

**Idea:** Treat each face image as a high-dimensional vector (each pixel is a dimension) and use PCA to find principal components (Eigenfaces).[1]

**Example :**[1]
- **Dataset:** 1000 faces, each 64×64 pixels = 4096 dimensions
- **PCA:** Find eigenfaces (eigenvectors of face covariance matrix)
- **Reconstruction:** Express any face as a combination of eigenfaces

**Algorithm Steps :**[1]

1. **Collect Face Dataset:**
   - Gather many face images (e.g., 1000 faces at 64×64 pixels)
   
2. **Flatten Images:**
   - Convert each 64×64 image into a 4096-dimensional vector

3. **Compute Mean Face:**
   - Average all face vectors to get the "mean face" (μ)

4. **Center Data:**
   - Subtract mean face from all faces

5. **Compute Covariance Matrix:**
   - Calculate covariance between pixel positions

6. **Find Eigenfaces:**
   - Compute eigenvectors of covariance matrix
   - These are the "eigenfaces"—basis face patterns

7. **Choose k Eigenfaces:**
   - Keep top k eigenfaces (e.g., 50, 100, 250)

8. **Represent Faces:**
   - Express each face as coordinates in eigenface space

9. **Reconstruction:**
   - Approximate original faces using eigenface combinations

***

#### **Reconstruction Examples from Lecture**

**Data Compression Impact :**[1]

1. **50 Eigenfaces:**
   - Original: 1000 × 64 × 64 = 4,096,000 values
   - Compressed: 50 × 64 × 64 = 204,800 values
   - **Compression:** 95% reduction
   - **Quality:** Recognizable but blurry

2. **100 Eigenfaces:**
   - Compression: 90% reduction
   - Quality: Better detail

3. **250 Eigenfaces:**
   - Compression: 75% reduction
   - Quality: Very good

4. **1000 Eigenfaces (all):**
   - Compression: None
   - Quality: Perfect reconstruction (all information retained)

***

#### **Dimensionality Trick**

**Challenge:** Computing eigenvectors of a 4096×4096 covariance matrix is computationally expensive.[1]

**Solution:** For N images of dimensionality d (where N < d), compute eigenvectors of the N×N matrix instead of d×d matrix.[1]

**Benefit:** 
- Instead of 4096×4096 matrix (for 64×64 images)
- Compute 1000×1000 matrix (for 1000 images)
- Massively reduces computation!

***

## **Summary**

This comprehensive guide has covered the essential concepts of feature selection and dimensionality reduction in machine learning :[1]

### **Feature Selection :**[1]
- **Purpose:** Select most relevant features, eliminate redundant/irrelevant ones
- **Search Strategies:** Exhaustive, Naive, Genetic Algorithms
- **Evaluation:** Filter methods (fast, algorithm-independent) vs. Wrapper methods (accurate, computationally expensive)
- **Sequential Methods:** SFS (bottom-up), SBS (top-down), LRS (limited backtracking), SFFS/SFBS (flexible backtracking), BDS (bidirectional)

### **Dimensionality Reduction :**[1]
- **Motivation:** Feature selection methods are expensive or suboptimal
- **PCA:** Most common technique—finds orthogonal directions of maximum variance
- **Statistical Foundation:** Mean, variance, covariance, eigenvectors/eigenvalues
- **Process:** Center data → Compute covariance matrix → Find eigenvectors → Project data
- **Component Selection:** Based on variance explained (typically 95-99%)

### **Applications :**[1]
- **Data Compression:** Reduce storage and computation
- **Visualization:** Project high-dimensional data to 2D/3D
- **Facial Recognition:** Eigenfaces technique
- **Preprocessing:** Remove noise before classification

***

## **Reading Materials**

For deeper understanding, consult these resources :[1]

1. **Chapter 5:** Feature Selection Methods and Results by Ali Hassan
2. **Parcel:** Feature Subset Selection in Variable Cost Domains (Cambridge Technical Report by M. Scott, M. Niranjan)
3. **A Tutorial on Principal Components Analysis** by Lindsay I Smith

***

## **Key Takeaways**

### **When to Use Feature Selection:**
- Have domain knowledge about which features matter
- Want interpretability (which features are selected?)
- Features are naturally discrete/categorical

### **When to Use PCA:**
- Have many correlated features
- Need maximum variance preservation
- Don't need interpretability of transformed features
- Want automatic, data-driven approach

### **Critical Distinctions:**
- **Feature Selection:** Keeps original features (subset)
- **PCA:** Creates new features (linear combinations)
- **PCA ≠ Linear Regression:** PCA has no target variable, minimizes perpendicular distance

### **Computational Considerations:**
- **Exhaustive search:** Infeasible for large feature sets
- **Wrapper methods:** Accurate but expensive (train many models)
- **Filter methods:** Fast but suboptimal
- **PCA:** Automated and efficient for dimensionality reduction

***

<div align="center">



</div>

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77906941/8945e5cf-43bd-4424-9d0a-b1e6793f539d/Pre-Lecture-6-Feature-Selection-Dimensionality-Reduction.pdf)
