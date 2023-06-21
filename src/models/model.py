import pickle
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, LeaveOneOut

SEED = 42


def train_model(features: pd.DataFrame, target: pd.Series) -> CatBoostRegressor:
    model = CatBoostRegressor(logging_level="Silent", random_state=SEED)
    transformer = ColumnTransformer(
        [
            (
                "numerical_pipeline",
                Pipeline(steps=[("scaler", MinMaxScaler())]),
                features.columns,
            ),
        ]
    )
    pipeline = Pipeline([("preprocessing_part", transformer), ("model_part", model)])

    cv = LeaveOneOut()
    score = cross_val_score(
        pipeline, features, target, cv=cv, scoring="neg_mean_absolute_percentage_error"
    )

    pipeline.fit(features, target)

    return pipeline, score


def save_model(model: object, output: str) -> str:
    with open(output, "wb") as model_file:
        pickle.dump(model, model_file)
    return output


def read_model(input: str) -> object:
    with open(input, "rb") as model_file:
        model = pickle.load(model_file)
    return model
