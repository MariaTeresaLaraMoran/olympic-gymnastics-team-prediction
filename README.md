# 🏅 Olympic Gymnastics Team Prediction Model

##  Project Overview

This project develops a machine learning model to predict team performance in elite gymnastics competitions, with the ultimate goal of simulating Olympic-level outcomes.

The model focuses on predicting routine scores and aggregating them into team scores to estimate country rankings for both:

* **Men’s Artistic Gymnastics (MAG)**
* **Women’s Artistic Gymnastics (WAG)**

---

## 🎯 Objectives

* Predict individual routine scores using historical competition data
* Simulate team totals based on predicted performances
* Generate realistic country rankings
* Understand the drivers of elite gymnastics performance


## 📊 Data

The dataset includes only meets from 2021-2023:
* Tokyo Olympics as a baseline for score tier performances
* International competitions (World Championships, World Cups, FISU university games)
* Regional competitions (Pan American, European, Asian, Commonwealth)
* Domestic competitions (USA Championships, Core Hydratation, Brithis championships)
* Athlete-level results by apparatus

### Key Variables

* Score (target)
* D-score (difficulty)
* E-score (execution)
* Competition, Round, Year
* Athlete, Country, Gender

##  Feature Engineering

The model integrates multiple layers of performance information:

### 1. Performance History. 

* **lag_1, lag_2, lag_3** → previous scores
* **avg_last_3** → rolling average

Captures **recent form and consistency**

### 2. Routine Composition (D & E Scores)

* **d_lag_1** → previous difficulty score
* **e_lag_1** → previous execution score
* **d_e_interaction = d_lag_1 × e_lag_1**

Models the balance between **difficulty and execution**

### 3. Team Depth ( only taking into consideration interantional meets)

* **depth_score** → weighted composition of Elite / High / Average athletes
* **elite_ratio** → proportion of elite athletes

Captures **team structure and stability**

### 4. Team Ceiling (Top-K Performance)

* **topk_score_weighted**
Computed using the top 3 athletes per country/apparatus:
* Best athlete → 50% weight
* Second → 30%
* Third → 20%

Represents the **maximum competitive potential of a team**

### 5. Olympic Strength

* **olympic_weight**

Based on Tokyo 2021:

* Medalists (×3)
* Finalists (×2)
* Participants (×1)

Captures **historical elite performance**

##  Machine Learning Models

📊 Data Split Strategy
Training: Domestic, regional, and international competitions (2022–2023)
Validation: 2023 international meets (group-based split to avoid athlete leakage)
Test: 2023 World Championships (true out-of-sample evaluation)

This approach ensures the model learns from diverse competition levels while being evaluated only on high-level international events, closely reflecting real-world prediction scenarios and preventing data leakage.

Two models were evaluated:
* **XGBoost (final model)**
* Multiple Linear Regression

Final Selection:
 **XGBoost** achieved the best performance.

##  Model Performance

| Model             | R² (Test) | RMSE     |
| ----------------- | --------- | -------- |
| XGBoost           |    0.47   | **0.80** |
| Linear Regression |    0.29   | 0.93     |
|

---

## 🏆 Team Simulation

Predicted routine scores are aggregated to simulate team totals.

## 🔍 Key Insights

* **WAG predictions are more stable** due to more consistent participation
* **MAG is more volatile**, with larger variation in performance
* **D-score plays a major role** in differentiating top teams
* **Team depth and ceiling both matter** for final rankings


## Future Improvements

* Add optimization and constraints about team structure,  to simulate Paris 2024 qualifiation and final round to get the team medals.

## 🧩 Tech Stack

* Python (Pandas, NumPy)
* Scikit-learn
* XGBoost
* Matplotlib (visualizations)

## 📌 Conclusion

This project demonstrates how combining:

* historical performance
* routine composition
* team structure
* and competition context

Can produce a realistic and interpretable model of elite gymnastics outcomes.
The final model balances predictive accuracy with interpretability, making it suitable for both analysis and decision support.


