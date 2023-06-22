import click
import logging
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from entities.train_pipeline_params import (
    read_train_pipeline_params,
    TrainPipelineParams,
)
from features.features import build_dataset
from models.model import (
    train_model,
    save_model,
)
from maps.map import plotting_map


def get_stream_handler() -> object:
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(log_format))
    return stream_handler


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(get_stream_handler())


@click.command()
@click.argument("config_path")
def train_pipeline(config_path: str) -> str:
    train_pipeline_params = read_train_pipeline_params(config_path)
    return run_train_pipeline(train_pipeline_params)


def run_train_pipeline(train_pipeline_params: TrainPipelineParams) -> str:
    logger.info(f"start train pipeline: {train_pipeline_params}")
    logger.info("build dataset")
    info, features, target = build_dataset(train_pipeline_params)
    logger.info(f"data shape: {features.shape}")

    logger.info("train model")
    pipeline, score = train_model(features, target["Выручка р/мес"].values)
    logger.info(f"CatBoostRegressor MAPE: {round(-np.mean(score), 3)}")

    logger.info("saving model")
    path_to_model = save_model(pipeline, train_pipeline_params.output_model_path)

    logger.info("start predict model")
    predicts = pipeline.predict(features)
    result = pd.concat(
        [
            info,
            target[["Выручка р/мес", "Чеки шт/мес"]],
            pd.DataFrame(predicts, columns=["Предсказание выручки"]),
        ],
        axis=1,
    )
    result.to_csv(train_pipeline_params.output_predict_path, index=False)

    logger.info("plotting shap")
    explainer = shap.TreeExplainer(pipeline["model_part"])
    transform_features = pd.DataFrame(
        pipeline["preprocessing_part"].transform(features), columns=features.columns
    )
    shap_values = explainer.shap_values(transform_features)
    shap.summary_plot(shap_values, transform_features, show=False)
    plt.savefig(train_pipeline_params.output_shap_path, dpi=100, bbox_inches="tight")

    if train_pipeline_params.plotting_map:
        logger.info("plotting map")
        plotting_map(train_pipeline_params, pipeline["model_part"], transform_features, result)
    return path_to_model


if __name__ == "__main__":
    train_pipeline()
