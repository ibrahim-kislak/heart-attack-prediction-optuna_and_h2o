# Heart disease risk prediction and non-anginal pain analysis

This project predicts heart attack risk from clinical data. It focuses on identifying high-risk patients who present with non-anginal pain—a symptom often misclassified in early triage. The model uses LightGBM, tuned with Optuna, and evaluates performance using H2O's Kolmogorov-Smirnov (KS) and Gains/Lift metrics to set operational hospital admission thresholds.

## Exploratory data analysis

The dataset contains records for 303 patients. The target variable is balanced, with 45.5% low-risk and 54.5% high-risk cases. We bypassed SMOTE and class weighting.

The chest pain (`cp`) variable shows an inverse relationship with expected clinical risk. Patients coded as `cp = 2` (non-anginal pain) have the highest rate of actual heart attacks in this sample. Typical angina (`cp = 0`) correlates with a lower risk of acute events, likely because these patients receive early medical intervention. The model relies heavily on the `cp = 2` variable to flag hidden risks.

Variables like age and maximum heart rate (`thalachh`) correlate predictably. We found no severe multicollinearity (all feature correlations sit below 0.70).

## Modeling strategy

We chose LightGBM for classification. Tree-based gradient boosting handles unscaled, mixed-type tabular data natively, so we did not normalize or standardize the input features.

## Hyperparameter optimization with Optuna

We used Optuna's Tree-structured Parzen Estimator (TPE) to tune the LightGBM classifier. The objective function maximized the F1-score to balance precision and recall. 

After 50 trials, Optuna selected a learning rate of 0.017 and 54 estimators. This constrained configuration prevents the model from overfitting the small 303-row dataset. The final validation F1-score reached 0.89.

## Evaluation using H2O and KS metrics

Standard confusion matrices obscure operational decisions. We used H2O to generate Gains, Lift, and Threshold tables to evaluate how the model performs as a resource allocation tool for a hospital.

**Kolmogorov-Smirnov (KS) statistic**
The model achieves a maximum KS statistic of 59.3% at a probability threshold of 0.667. This threshold maximizes the mathematical separation between the healthy and high-risk patient distributions.

**Gains and capture rate**
Sorting patients by predicted risk shows the practical impact of that threshold. Admitting the top 60.6% of patients captures 87.8% of all actual heart attacks. The response rate within this admitted group is 78.3%. Hospitals can use this threshold to allocate ICU beds to the top 60% of cases, securing nearly 90% of critical patients without running extensive tests on the entire population.

## Setup and installation

Install the required packages to run the notebook:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm optuna h2o
