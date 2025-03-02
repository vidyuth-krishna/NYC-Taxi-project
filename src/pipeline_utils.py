import lightgbm as lgb
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import RandomizedSearchCV

# Function to calculate the average rides over the last 4 weeks
def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    last_4_weeks_columns = [
        f"rides_t-{7*24}",  # 1 week ago
        f"rides_t-{14*24}",  # 2 weeks ago
        f"rides_t-{21*24}",  # 3 weeks ago
        f"rides_t-{28*24}",  # 4 weeks ago
    ]

    # Ensure the required columns exist in the DataFrame
    for col in last_4_weeks_columns:
        if col not in X.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate the average of the last 4 weeks
    X["average_rides_last_4_weeks"] = X[last_4_weeks_columns].mean(axis=1)

    return X


# FunctionTransformer to add the average rides feature
add_feature_average_rides_last_4_weeks = FunctionTransformer(
    average_rides_last_4_weeks, validate=False
)


# Custom transformer to add temporal features
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_["hour"] = X_["pickup_hour"].dt.hour
        X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek

        return X_.drop(columns=["pickup_hour", "pickup_location_id"])


# Instantiate the temporal feature engineer
add_temporal_features = TemporalFeatureEngineer()

# hyper_params = {
#     "n_estimators": 100,
#     "learning_rate": 0.05,
#     "max_depth": 5
#     }

# Function to return the pipeline
param_distributions = {
    "lgbmregressor__num_leaves": [2, 50, 70, 256],
    "lgbmregressor__max_depth": [-1, 10, 20, 30],
    "lgbmregressor__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "lgbmregressor__n_estimators": [100, 200, 500, 1000],
    "lgbmregressor__min_child_samples": [10, 20, 30, 50],
    "lgbmregressor__subsample": [0.6, 0.8, 1.0],
    "lgbmregressor__colsample_bytree": [0.6, 0.8, 1.0],
    "lgbmregressor__reg_alpha": [0, 0.1, 0.5, 1.0],
    "lgbmregressor__reg_lambda": [0, 0.1, 0.5, 1.0],
    "lgbmregressor__feature_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
    "lgbmregressor__bagging_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
    "lgbmregressor__bagging_freq": [1, 5, 10]
}

def get_pipeline(use_random_search=False, **hyper_params):
    """
    Returns a pipeline with optional hyperparameter tuning using RandomizedSearchCV.

    Parameters:
    ----------
    use_random_search : bool
        Whether to return a RandomizedSearchCV object.
    
    **hyper_params : dict
        Optional parameters to pass to LGBMRegressor.

    Returns:
    -------
    model : sklearn.pipeline.Pipeline or RandomizedSearchCV
        The pipeline or the hyperparameter-tuned model.
    """
    pipeline = make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(verbose=-1, **hyper_params)
    )

    if use_random_search:
        model = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=10,
            scoring="neg_mean_absolute_error",
            cv=3,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = pipeline

    return model
