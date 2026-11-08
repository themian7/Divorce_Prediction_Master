## Marriage Divorce Prediction — Machine Learning Project

## Project Overview

The goal of this project is to predict whether a marriage will end in divorce using demographic, financial, and relationship-related information.

The dataset contains features such as conflict frequency, number of children, financial stress, relationship behavior, and marital history.

This is a binary classification problem, where the target variable "divorced" represents whether a marriage ended in divorce.

## Dataset

The project uses a publicly available CSV dataset from Kaggle.

The dataset contains approximately 45,000 records and includes both numerical and categorical features.

Dataset source: [https://sl1nk.com/tu364i4]

The objective is to investigate whether the available information can help predict divorce outcomes.

## Exploratory Data Analysis (EDA)

Before training any models, I explored the dataset to understand its structure, quality, and feature distributions.

The analysis included:

- Checking missing values, duplicates, and potentially invalid observations.
- Examining numerical and categorical features.
- Visualizing feature distributions using histograms and boxplots.
- Investigating relationships between features and the target variable.
- Using Pandas operations such as "groupby()" and "crosstab()" to support observations from visualizations.

Skewness and Outlier Analysis

Several numerical features were right-skewed and contained numerous outliers.

After examining these observations, I found no clear evidence that the unusual values were invalid. Therefore, I retained them rather than automatically removing or replacing them.

Potential Data Leakage

I excluded "years_to_divorce" because it represents information that would only become available after a divorce occurred.

I also excluded "years_married" from the final model because its interpretation could make it unsuitable for the intended prediction scenario.

Including "years_married" previously produced approximately 78% accuracy, while excluding it reduced performance to approximately 75%.

This highlighted an important lesson: a higher accuracy score is not necessarily better if the model relies on information that would not be available at prediction time.

## Data Preprocessing

The preprocessing stage included:

- Separating features and the target variable.
- Splitting the dataset into training and test sets.
- Encoding categorical features.
- Standardizing numerical features for scale-sensitive algorithms.
- Excluding features identified as potentially unsuitable or likely to introduce leakage.

## Model Training and Comparison

I started with Logistic Regression as the baseline model and then experimented with several other classification algorithms.

Initial Model Results

Model| Test Accuracy
Logistic Regression| 75.31%
Support Vector Classifier| 74.83%
Gaussian Naive Bayes| 72.86%
K-Nearest Neighbors| 69.28%
Decision Tree| 64.31%

Logistic Regression achieved the highest accuracy among the initial models.

Ensemble Learning

I also experimented with ensemble methods to investigate whether combining models or using more complex algorithms could improve performance.

Model| Test Accuracy
XGBoost| 74.9%
Gradient Boosting| 74.9%
Stacking Classifier| 74.9%
Random Forest| 73.6%

The ensemble methods did not produce a meaningful improvement over Logistic Regression.

This demonstrated that greater model complexity does not automatically result in better predictive performance.

## Cross-Validation

I used 5-fold cross-validation to assess model performance across different validation subsets.

Model| Mean CV Accuracy
Logistic Regression| 75.16%
Support Vector Classifier| 75.01%
Gaussian Naive Bayes| 72.50%
K-Nearest Neighbors| 69.20%
Decision Tree| 64.65%

Logistic Regression remained the strongest candidate based on mean cross-validation accuracy.

The similarity between its cross-validation and test results provided additional confidence in its performance consistency.

## Hyperparameter Tuning

I used "GridSearchCV" to tune Logistic Regression.
- "C = 100"
- "solver = lbfgs"

The best cross-validation accuracy was approximately 75.16%.

Hyperparameter tuning did not produce a meaningful improvement over the original model.

An extensive SVC grid search was also attempted but discontinued because of its computational cost.

## Feature Engineering

I experimented with additional features derived from existing variables, including:

- "total_negativity": a combination of negative relationship-behavior indicators.
- "income_per_person": household income adjusted for household size.
- "conflict_x_repair": an interaction between conflict frequency and repair-attempt success.

These engineered features produced only minor changes in model performance, with some algorithms improving slightly and others performing marginally worse.

This suggested that the original features already captured much of the information represented by these combinations.

## Final Model Evaluation

Logistic Regression was selected as the final model based on its overall performance and relative simplicity.

The tuned model achieved approximately 75% test accuracy.

Its confusion matrix was:

| Predicted: No Divorce| Predicted: Divorce
Actual: No Divorce| 3,870| 1,004
Actual: Divorce| 1,214| 2,912

The model correctly identified many outcomes but still made both false-positive and false-negative predictions.

Therefore, accuracy alone should not be treated as a complete measure of predictive performance. Precision, recall, and F1-score are also important.

## Key Learnings

This project helped me to understand several important aspects of practical machine learning.

1. Complex models are not always better.

Despite experimenting with multiple algorithms and ensemble methods, Logistic Regression remained the strongest candidate.

2. Data leakage can create misleading results.

Features must be evaluated based on whether their information would actually be available when predictions are made.

3. Cross-validation helps assess model stability.

It provides a more reliable basis for comparing models than relying on a single validation result.

4. Hyperparameter tuning does not guarantee improvement.

Even after searching different configurations, model performance remained almost unchanged.

5. Feature engineering requires meaningful reasoning.

Creating additional features does not necessarily introduce new predictive information.

6. Valid outliers do not automatically require removal.

Unusual observations should be investigated before deciding whether to transform or exclude them.

7. Conclusion

Logistic Regression achieved approximately 75% accuracy and performed slightly better than the other tested algorithms.

Neither more complex ensemble methods nor the attempted feature engineering produced substantial improvements.

The project demonstrated that model selection, data quality, leakage prevention, validation, and interpretation are just as important as algorithm complexity.

## Limitation
The dataset's representativeness and suitability for real-world predictions have not been established. Therefore, the results should be treated as an educational machine learning experiment rather than a validated system for predicting real marriages.

## How to Run
1.Install Python (I used 3.13) and the libraries: pip install -r requirements.txt

2.Download the dataset from the Kaggle link given above and put marriage_longevity_master.csv next to the notebook.

3.Open the notebook in Jupyter or VS Code and run all cells.

Tools: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost

## Target Feature
![Target Feature](images\target_feature.png)

## Confusion Matrix of Logistic Regression
![Target Feature](images\confusion_matrix_lr.png)

## Model Comparison of Cross Validation
![Target Feature](images\cross_validation.png)
