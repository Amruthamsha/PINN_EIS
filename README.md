# ⚖️ Physics-Informed Machine Learning for Battery SoC
**Author:** Amruthamsha P Raju  

---

## 📘 Overview
This project focuses on strategies to handle **imbalanced datasets** in machine learning.  
Class imbalance occurs when certain classes are underrepresented, which can cause models to be biased towards the majority class. Effective handling of imbalance improves model fairness and predictive performance.

---

## 🧠 Core Concepts
- **Class Imbalance:** When one class has significantly fewer samples than others.  
- **Challenges:**  
  - Biased predictions toward majority class  
  - Poor generalization for minority class  
  - Reduced overall model performance  

- **Common Strategies:**  
  1. **Data-Level Techniques:**  
     - Oversampling minority class (e.g., SMOTE)  
     - Undersampling majority class  
     - Synthetic data generation  
  2. **Algorithm-Level Techniques:**  
     - Cost-sensitive learning (higher penalty for misclassifying minority)  
     - Ensemble methods (Balanced Random Forest, EasyEnsemble)  
  3. **Hybrid Techniques:** Combine both data- and algorithm-level approaches  

---

## ⚙️ Workflow
1. **Load and explore dataset**  
2. **Analyze class distribution**  
3. **Apply imbalance handling techniques**  
4. **Train machine learning models**  
5. **Evaluate using balanced metrics:**  
   - F1 Score  
   - Precision & Recall  
   - ROC-AUC  

---

## 🧪 Evaluation Metrics
- **Accuracy**: May be misleading for imbalanced data  
- **F1 Score**: Harmonic mean of precision and recall, better for minority class  
- **Balanced Accuracy**: Accounts for class distribution  
- **ROC-AUC**: Measures separability between classes  

---

## 🧭 Goals
- Improve predictive performance for minority classes  
- Reduce model bias towards majority class  
- Demonstrate practical strategies for real-world imbalanced datasets  

---

## 🧾 References
- Chawla, N. V., et al. “SMOTE: Synthetic Minority Over-sampling Technique.” *Journal of Artificial Intelligence Research*, 2002.  
- He, H., & Garcia, E. A. “Learning from Imbalanced Data.” *IEEE Transactions on Knowledge and Data Engineering*, 2009.  
