from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TrainPipelineParams:
    input_target_path: str
    input_data_path: str
    input_isochrone_10_path: str
    input_isochrone_15_path: str
    input_isochrone_20_path: str
    input_isochrone_25_path: str
    input_isochrone_30_path: str
    input_distance_path: str
    input_highway_path: str
    input_reestr_path: str
    output_model_path: str
    output_shap_path: str
    output_predict_path: str
    output_map_path: str
    plotting_map: bool = field(default=False)


TrainPipelineParamsSchema = class_schema(TrainPipelineParams)


def read_train_pipeline_params(path: str) -> TrainPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
