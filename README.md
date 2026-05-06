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

### 4. Olympic Strength

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

| Model | Train RMSE | Test RMSE | Train R² | Test R² |
|---|---:|---:|---:|---:|
| XGBoost | 0.952937 | 0.788155 | 0.562402 | 0.488843 |
| Linear Regression | 1.175697 | 0.875863 | 0.333904 | 0.368747 |
---

## 🏆 Team Simulation

## Final Team Ranking Simulation: 2023 World Championships

To evaluate the model in a realistic competition scenario, I compared the predicted team rankings against the actual final team rankings from the 2023 World Championships. The simulation used the top 8 qualified teams and compared the predicted team score with the actual team final result.

This comparison helps show whether the model was able to preserve the competitive order of the teams, even when the exact predicted scores were not perfect.

### MAG Team Final Results

| Country | Predicted Score | Actual Score | Score Difference | Absolute Error | Predicted Rank | Actual Rank | Rank Difference |
### MAG Team Final Results

| Country | Predicted Score | Actual Score | Score Difference | Absolute Error | Predicted Rank | Actual Rank | Rank Difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| JPN | 253.21 | 255.59 | -2.39 | 2.39 | 1 | 1 | 0 |
| CHN | 252.75 | 253.79 | -1.04 | 1.04 | 2 | 2 | 0 |
| USA | 251.19 | 252.43 | -1.23 | 1.23 | 3 | 3 | 0 |
| GBR | 247.46 | 249.46 | -2.01 | 2.01 | 4 | 4 | 0 |
| SUI | 239.66 | 244.43 | -4.77 | 4.77 | 7 | 5 | 2 |
| GER | 239.52 | 244.03 | -4.51 | 4.51 | 8 | 6 | 2 |
| CAN | 240.13 | 243.03 | -2.90 | 2.90 | 6 | 7 | -1 |
| ITA | 244.64 | 241.16 | 3.48 | 3.48 | 5 | 8 | -3 |

### WAG Team Final Results

| Country | Predicted Score | Actual Score | Score Difference | Absolute Error | Predicted Rank | Actual Rank | Rank Difference |
### WAG Team Final Results

| Country | Predicted Score | Actual Score | Score Difference | Absolute Error | Predicted Rank | Actual Rank | Rank Difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| USA | 166.72 | 167.73 | -1.01 | 1.01 | 1 | 1 | 0 |
| BRA | 161.56 | 165.53 | -3.97 | 3.97 | 3 | 2 | 1 |
| FRA | 157.99 | 164.06 | -6.07 | 6.07 | 8 | 3 | 5 |
| CHN | 163.06 | 163.16 | -0.10 | 0.10 | 2 | 4 | -2 |
| ITA | 159.86 | 163.00 | -3.14 | 3.14 | 4 | 5 | -1 |
| GBR | 159.78 | 161.86 | -2.09 | 2.09 | 5 | 6 | -1 |
| NED | 158.97 | 159.56 | -0.59 | 0.59 | 6 | 7 | -1 |
| JPN | 158.41 | 157.50 | 0.91 | 0.91 | 7 | 8 | -1 |

### Interpretation

The MAG simulation performed strongly in terms of ranking accuracy. The model correctly predicted the top four teams in the exact same order as the actual 2023 World Championships team final results: Japan, China, USA, and Great Britain. This explains the strong Spearman correlation for MAG.

The WAG simulation also captured part of the ranking structure, especially by correctly identifying USA as the top team. However, the model had more difficulty with the middle-ranked teams. France had the largest rank difference, finishing third in the actual final but being predicted eighth. This suggests that the WAG model may need stronger features for recent team improvement, lineup changes, execution consistency, and competition-specific performance.

Overall, these final tables show that the model was more reliable for ranking structure than for exact score prediction, especially for MAG. This is important in gymnastics because competition outcomes depend heavily on relative placement, consistency, and small differences in execution.

## Spearman Rank Correlation Analysis

At the end of the project, I calculated the **Spearman rank correlation** between the predicted team rankings and the actual team rankings for MAG and WAG.

Spearman correlation is important for this project because gymnastics is not only about predicting the exact final score, but also about predicting the relative ranking of teams. In a judged sport, very small score differences can change the final placement, so rank-based evaluation gives an additional perspective beyond RMSE.

### Results

| Group | Spearman Correlation | P-value |
|---|---:|---:|
| MAG | 0.786 | 0.02082 |
| WAG | 0.595 | 0.11953 |

### Interpretation

For **MAG**, the Spearman correlation was **0.786** with a **p-value of 0.02082**. This suggests a strong positive relationship between the predicted rankings and the actual rankings. Since the p-value is below 0.05, the result is statistically significant.

For **WAG**, the Spearman correlation was **0.595** with a **p-value of 0.11953**. This suggests a moderate positive relationship between the predicted rankings and the actual rankings, but the result is not statistically significant at the traditional 0.05 level. This means the WAG ranking result should be interpreted more cautiously. One possible reason WAG was harder to predict is the structure of the team final itself. In the team final format, only three athletes compete on each apparatus and all three scores count. This makes the result less forgiving because a single fall or major execution error can significantly change the team ranking.

This is especially important in WAG because balance beam can introduce high variability. Beam routines are more likely to create large score swings when athletes fall or have major balance breaks. Because these moments are difficult to predict from historical averages alone, the model had more difficulty capturing the final WAG ranking structure.

In addition, WAG receives very high public attention compared with many other gymnastics categories, which may increase competitive pressure in major finals. However, this project does not directly measure psychological pressure, so this should be interpreted as a possible explanation rather than a confirmed model feature.

Overall, these results suggest that the model captured ranking structure better for MAG than WAG. The WAG result still shows a positive relationship, but the larger rank error for France affected the overall correlation and p-value.


## 🔍 Key Insights

* **The model captured ranking patterns better than exact scores.** This was especially clear in the 2023 Worlds team final simulation, where the predicted order was more informative than the raw score error.
* **MAG showed stronger rank prediction in the final simulation**, with a Spearman correlation of 0.762 and the top four teams predicted in the correct order.
* **WAG showed a positive ranking relationship**, but with more uncertainty. The model correctly identified USA as the top team, but had larger rank differences for teams such as France and China.
* **D-score plays a major role** in differentiating top teams, especially when comparing teams with similar execution levels.
* **Team depth and elite ratio matter** because the final team score depends not only on one star athlete, but on how many gymnasts can contribute competitive scores across events.
* **Judged sport variability remains an important limitation.** Execution scores, falls, lineup decisions, and judging differences can create changes that are difficult to predict from historical data alone.


## Future Improvements

* Add optimization constraints for team structure in order to simulate qualification and final rounds more realistically.
* Build a lineup optimization system that selects the best team score under real gymnastics rules, such as team size, number of athletes per apparatus, and counting scores.
* Simulate Paris 2024 qualification and team final medal outcomes using different team selection strategies by forecasting the improvement of the routines over time.
* Add uncertainty intervals around predicted team scores to reflect real competition variability.
* Include additional features related to athlete consistency, hit rate, fall risk, and recent performance trends.
* Explore classification models for predicting whether a routine is likely to be a “hit” or “no hit.”

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
* athlete depth
* and competition context

Can produce a realistic and interpretable model of elite gymnastics outcomes.

The final model was more successful at capturing relative team rankings than predicting exact scores. This is important because gymnastics outcomes are determined by placement, and small score differences can change the final ranking. The Spearman rank correlation and 2023 Worlds simulation showed that the model can identify meaningful ranking patterns, especially for MAG.

At the same time, the project also highlights the difficulty of modeling judged sports. Execution quality, falls, judging variation, athlete mental and physical health, and lineup decisions introduce uncertainty that cannot be fully captured with historical score data alone. Because of this, the model should be interpreted as an analytical simulation tool rather than a perfect prediction system.

