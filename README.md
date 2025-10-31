# Physics-Informed Machine Learning for Battery State of Charge (SoC)
**Author:** Amruthamsha P Raju  

---

## 📘 Overview
This project implements a **Physics-Informed Neural Network (PINN)** to estimate the **State of Charge (SoC)** of batteries using **Electrochemical Impedance Spectroscopy (EIS)** data.  
Unlike purely data-driven models, PINNs integrate **physical knowledge** (impedance–SoC relationships) directly into the loss function, ensuring predictions are both **accurate and physically consistent**.

---

## 🧠 Key Concepts

- **Physics-Informed Learning:** Combines traditional regression objectives with **physics-based constraints**, e.g.,  
  - Negative correlation between impedance magnitude (|Z|) and SoC  
  - Smoothness of SoC predictions across adjacent points  

- **Custom Loss Function:**
- total_loss = MSE + 0.1 * correlation_penalty + 0.01 * smoothness_penalty

- **MSE:** Ensures data fidelity  
- **Correlation Penalty:** Enforces physical relationship between |Z| and SoC  
- **Smoothness Penalty:** Reduces unrealistic oscillations in predicted SoC  

- **Feature Engineering:**  
Input features include `[FREQUENCY_VALUE, ReZ, ImZ, |Z|]` derived from complex impedance measurements.

- **Benchmarking:** Classical ML regressors (Random Forest, Gradient Boosting, XGBoost) are used for comparison to highlight the added value of physics-informed constraints.

---

## ⚙️ Workflow

1. **Data Preprocessing:**  
 - Merge impedance and frequency datasets  
 - Extract real (`ReZ`) and imaginary (`ImZ`) components  
 - Compute impedance magnitude `|Z|`

2. **Train-Test Split:**  
 - 80/20 train-test split to evaluate generalization  

3. **PINN Construction:**  
 - Sequential neural network: Dense layers `[128, 128, 64]` with `tanh` activation  
 - Physics-informed loss integrated via a custom Keras loss class  

4. **Model Training:**  
 - Optimizer: Adam, learning rate = 1e-3  
 - Epochs: 200, Batch size: 32  

5. **Evaluation Metrics:**  
 - Regression: `MSE`, `RMSE`, `R²`  
 - Classification (high vs. low SoC): `Accuracy`, `F1 Score`  

6. **Comparison:**  
 - Classical ML models trained on the same features to assess performance improvement through physics integration

---

## 📊 Key Advantages

- Enforces **physically consistent predictions**, reducing non-physical anomalies in SoC estimates  
- Improves **generalization** with limited or noisy data  
- Combines **domain knowledge and data-driven learning** seamlessly  

---

## 🧭 Future Directions

- Incorporate **battery aging and temperature effects** for more robust SoC estimation  
- Extend to **time-series prediction** for real-time SoC tracking  
- Explore **Physics-Informed Transformers** for complex battery systems  

---

## 🧾 References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.” *Journal of Computational Physics*, 2019.  
- Chawla, N. V., et al. “SMOTE: Synthetic Minority Over-sampling Technique.” *Journal of Artificial Intelligence Research*, 2002.  
- He, H., & Garcia, E. A. “Learning from Imbalanced Data.” *IEEE Transactions on Knowledge and Data Engineering*, 2009.  
