# 🛒 Purchase Prediction using Decision Tree Classifier

## 📌 Project Overview

This project focuses on predicting whether a customer will purchase a product based on **Gender, Age, and Salary** using a **Decision Tree Classifier**. The solution follows the **complete Machine Learning Life Cycle**, including data understanding, exploratory data analysis (EDA), preprocessing, model training, validation, and inference.

This is a **classification problem** commonly used in marketing analytics and customer targeting use cases.

---

## 🎯 Business Objective

To help businesses:

* Identify potential customers
* Improve targeted marketing campaigns
* Reduce advertising costs
* Increase conversion rates

---

## 📊 Dataset Description

| Column Name | Description                       |
| ----------- | --------------------------------- |
| Gender      | Customer gender (Male/Female)     |
| Age         | Customer age                      |
| Salary      | Estimated annual salary           |
| Purchased   | Target variable (0 = No, 1 = Yes) |

---

## 🧠 Machine Learning Life Cycle

1. Business Understanding
2. Data Understanding
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
5. Model Building
6. Model Evaluation
7. Inference & Business Insights

---

## 🔍 Exploratory Data Analysis (EDA)

### Key Analysis Performed

* Purchase distribution analysis
* Gender vs Purchase behavior
* Age distribution and its impact on purchase
* Salary distribution and purchase correlation

### 📌 EDA Insights

* **Age** is a strong predictor of purchase behavior
* Users with **higher salaries** are more likely to purchase
* **Gender has minimal influence** compared to Age and Salary

Visualizations were created using **Seaborn and Matplotlib**.

---

## ⚙️ Data Preprocessing

* Converted categorical variable `Gender` into numeric format
* Split data into features (X) and target (y)
* Applied train-test split

```python
Gender → Male = 1, Female = 0
```

---

## 🌳 Model Building

### Algorithm Used

**Decision Tree Classifier**

### Why Decision Tree?

* Easy to interpret
* Handles non-linear relationships
* No feature scaling required
* Suitable for business explanations

```python
DecisionTreeClassifier(criterion='gini', max_depth=5)
```

---

## 📈 Model Evaluation

### Metrics Used

* Accuracy Score
* Confusion Matrix
* Precision, Recall, F1-score

### Performance Summary

* Achieved good classification accuracy
* Balanced performance across both classes
* Minimal overfitting due to controlled tree depth

---

## 🌲 Decision Tree Visualization

The trained decision tree was visualized to:

* Understand feature splits
* Explain model logic to non-technical stakeholders

Key features influencing decisions:

1. Age
2. Salary
3. Gender

---

## 🧾 Final Inference

* Customers aged **30+ with higher salary** are more likely to purchase
* Decision Tree effectively captures customer behavior
* Model is interpretable and business-friendly

---

## 🚀 Business Impact

* Enables targeted advertising
* Improves marketing ROI
* Helps identify high-value customers
* Reduces unnecessary ad spend

---

## ⚠️ Limitations

* Decision Trees can overfit if not tuned
* Performance may vary with noisy data

---

## 🔧 Future Improvements

* Use Random Forest or Gradient Boosting
* Perform hyperparameter tuning
* Add more customer features

---

## 🛠️ Tools & Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📁 Project Structure

```text
├── data/
│   └── dataset.csv
├── notebooks/
│   └── decision_tree_purchase_prediction.ipynb
├── graphs/
│   └── eda_visualizations.png
├── README.md
```

---

## 📌 Keywords (for GitHub SEO)

Decision Tree Classifier, Purchase Prediction, Machine Learning, Classification, EDA, Scikit-learn, Data Science Project, Marketing Analytics

---

## 👤 Author

**Nikhil Jaiswal**
Data Analyst | Machine Learning Enthusiast

---

⭐ *If you find this project useful, feel free to star the repository!*
