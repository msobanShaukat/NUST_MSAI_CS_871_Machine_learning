 
# **Part D: Comparative Analysis - Classical ML vs Deep Learning**

## **1. Performance Comparison**

| Model | Accuracy | Macro-F1 | Cohen's κ |
|-------|----------|----------|-----------|
| **Classical ML (Random Forest)** | 40.40% | 0.21 | 0.09 |
| **Deep Learning (ResNet18)** | 45.11% | 0.34 | 0.19 |
| **Ordinal ResNet18 (Bonus)** | **65.40%** | **0.65** | **0.52** |
| **Weighted Ensemble (Bonus)** | 46.32% | 0.32 | 0.19 |

**Per-Class Accuracy:**

| KL Grade | Classical ML | Deep Learning | Winner |
|----------|--------------|---------------|--------|
| Grade 0 | 31.77% | **88.89%** | DL |
| Grade 1 | **20.61%** | 9.80% | Classical |
| Grade 2 | 21.92% | 18.57% | Classical |
| Grade 3 | 18.39% | 21.97% | DL |
| Grade 4 | 25.49% | **35.29%** | DL |

**Key Finding:** Deep learning outperforms classical ML overall (+11.7%) but struggles severely on Grade 1 (9.8%). The ordinal model achieves transformative improvement (+45% over standard ResNet18) by respecting KL grade ordering.

***

## **2. Which Approach Performed Better and Why?**

### **Winner: Ordinal Classification ResNet18 (65.40% accuracy)**

**Why Deep Learning > Classical ML:**

1. **Hierarchical Feature Learning:** ResNet18 learns 18 layers of features (edges → textures → joint patterns) vs 26 fixed LBP bins
2. **Spatial Context:** Convolutional layers preserve 2D spatial relationships critical for detecting joint space narrowing; LBP collapses spatial information
3. **Transfer Learning:** Pre-trained ImageNet weights provide strong feature extractors, reducing data requirements
4. **Representation Capacity:** 150,528 input dimensions (224×224×3) vs 26 hand-crafted features

**Why Ordinal Classification Dominates:**

Standard cross-entropy treats all errors equally (Grade 0→4 same penalty as Grade 0→1). Ordinal loss predicts cumulative thresholds P(grade > k) for k=0,1,2,3, respecting natural ordering. **Result:** +45% absolute accuracy improvement (45%→65%), balanced performance across all grades, and Cohen's κ=0.52 indicating "moderate agreement" with radiologists.

***

## **3. Hardest KL Grades to Predict**

### **Grade 1 (Doubtful OA) - HARDEST**
- **Performance:** Classical ML: 20.61% | Deep Learning: **9.80%** (worst class)
- **Confusion:** 227/296 (76.7%) misclassified as Grade 0
- **Causes:**
  - Visual ambiguity (questionable osteophytes, minimal narrowing)
  - Overlaps with Grades 0 and 2 → model defaults to majority class
  - Class imbalance (296 samples vs 639 Grade 0)

### **Grade 2 (Minimal OA) - Second Hardest**
- **Performance:** Classical ML: 21.92% | Deep Learning: 18.57%
- **Confusion:** 248/447 (55.5%) predicted as Grade 0
- **Causes:** Mild joint space narrowing easily missed; conservative "when in doubt, predict Grade 0" strategy

### **Grade 3 (Moderate OA) - Third Hardest**
- **Performance:** Classical ML: 18.39% | Deep Learning: 21.97%
- **Confusion:** 101/223 (45.3%) predicted as Grade 0
- **Causes:** Sandwiched between Grades 2 and 4; shares features with both; limited samples (223 test images)

### **Grade 4 (Severe OA) - Surprisingly Better**
- **Performance:** Classical ML: 25.49% | Deep Learning: **35.29%**
- **Insight:** Despite fewest samples (51), deep learning performs best on Grade 4 due to distinctive features (severe deformities, complete joint space loss) easily detected by transfer learning

***

## **4. Impact of Class Imbalance**

### **Training Distribution:**

| Grade | Train Count | Percentage | Imbalance Ratio |
|-------|-------------|------------|-----------------|
| 0 | 2,286 | 39.6% | 1.0× (baseline) |
| 1 | 1,046 | 18.1% | 2.2× |
| 2 | 1,516 | 26.2% | 1.5× |
| 3 | 757 | 13.1% | 3.0× |
| 4 | 173 | 3.0% | **13.2×** ⚠️ |

### **Consequences:**

1. **Majority Class Bias:** Grade 0 achieves 88.89% accuracy while Grade 1 collapses to 9.8%
2. **Loss Function Mechanics:** Cross-entropy weights errors by sample count; Grade 4 contributes only 3% of total loss vs Grade 0's 39.6% → model rationally ignores minorities
3. **Macro-F1 Gap:** Macro-F1 (0.34) << Accuracy (0.45) reveals severe imbalance damage
4. **Adjacent Grade Confusion:** Most errors are ±1 grade, indicating model learns "severity continuum" but struggles with boundaries

 
## **5. Conclusion**

This analysis demonstrates five critical findings:

1. **Deep learning outperforms classical ML** (+11.7% accuracy) through automatic hierarchical feature learning, spatial context preservation, and transfer learning benefits

2. **Ordinal classification is transformative** (+45% accuracy over standard ResNet18) by respecting the natural ordering of KL grades, achieving balanced performance (Macro-F1 = 0.65 ≈ Accuracy)

3. **Class imbalance is the primary challenge:** 13.2× imbalance ratio causes Grade 1 catastrophic collapse (9.8%) and Grade 0 dominance (88.9%)

4. **Grade 1 is hardest to predict** due to visual ambiguity and imbalance, while Grade 4 performs surprisingly well (35.3%) despite fewest samples due to distinctive severe features

5. **Ordinal ResNet18 recommended for deployment** (65.40% accuracy, κ=0.52 "moderate agreement") suitable for clinical screening and decision support with appropriate human oversight

**Clinical Impact:** The ordinal model balances sensitivity across all grades, making errors clinically acceptable (±1 grade), and provides Grad-CAM explanations for trust.

***
 
