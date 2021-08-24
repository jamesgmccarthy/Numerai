# Numerai
Models for Numerai Competition

Ensemble of 3 model types (LightGBM, CatBoost & XGBoost), trained through 10 fold Purged Group Time Series Split cross-validation (to ensure no data leakage during training), three times each with different seeds.
Each model is saved and then each is used to predict on the test set.
The mean of each model type's predictions are computed and the mean of the 3 is computed and used as the final submission.
Perhaps some optimization could be used to best weight the means but for now its a simple fully weighted mean. 

