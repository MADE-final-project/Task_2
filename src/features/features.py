import pandas as pd
from entities.train_pipeline_params import TrainPipelineParams


WORK_SCHEDULE = [
    ["с 9-00 до 22-00", 9, 22, 9, 22, 0],
    ["с 8-00 до 22-00", 8, 22, 8, 22, 0],
    ["пн-чт с 7-00 - 23-45, пт-сб с 8-00 - 23-00, вс с 7-00 - 23-45", 7, 23, 7, 23, 1],
    ["Откл. с 9:00 до 22:00", 9, 22, 9, 22, 0],
    ["с 7-00 до 22-00", 7, 22, 7, 22, 0],
    ["будни с 8-00 до 22-00; вых. с 9-00 до 22-00", 8, 22, 9, 22, 1],
    ["с 10-00 до 22-00", 10, 22, 10, 22, 0],
    ["будни с 9-00 до 23-00, выходные с 9-00 до 22-00", 9, 23, 9, 22, 1],
    ["с 9-00 до 21-00", 9, 21, 9, 21, 0],
    ["Будни с 8 до 23, вых с 8 до 22", 8, 23, 8, 22, 1],
    ["с 7:00 до 23:45", 7, 23, 7, 23, 0],
    ["Откл. с 11:00 до 21:00", 11, 21, 11, 21, 0],
    ["Откл. с 8:00 до 22:00", 8, 22, 8, 22, 0],
    ["с 8-00 до 23-45", 8, 23, 8, 23, 0],
    ["с 8-00 до 23-00", 8, 23, 8, 23, 0],
    ["Будни с8:00 до 23:00 вых с 8:00 до 22:00", 8, 23, 8, 22, 1],
    ["будни с 8-00 до 23-00, вых. 9-00 до 23-00", 8, 23, 9, 23, 1],
    ["пн-пт с 8-00 до 23-00, сб-вс с 9-00 до 22-00", 8, 23, 9, 22, 1],
    ["вс-чт. с 10:00 до 23:00, пт-сб. с 10:00 по 24:00", 10, 23, 10, 24, 1],
    ["пн-чт с 7-00 - 23-45, пт-сб с 7-00 - 23-00, вс с 7-00 - 23-45", 7, 23, 7, 23, 1],
    ["с 8-30 до 23-00", 8, 23, 8, 23, 0],
    ["с 7-00 до 23-45", 7, 23, 7, 23, 0],
    ["пн-чт с 7-00 - 23-45, пт-сб с 7-00 - 22-00, вс с 8-00 - 23-45", 7, 23, 8, 23, 1],
    ["с пн по чт с 7 до 23:45, пт с 7 до 23:00, сб с 8 до 22:00, вс 8 до 23:45", 7, 23, 8, 23, 1],
    ["бд 7-00 до23-00 вх 8-00 до 22-00", 7, 23, 8, 22, 1],
    ["с 9:00 до 23:00", 9, 23, 9, 23, 0],
    ["с 7-00 до 23-00", 7, 23, 7, 23, 0],
]


def build_dataset(train_pipeline_params: TrainPipelineParams):
    target = pd.read_csv(train_pipeline_params.input_target_path)
    data = pd.read_csv(train_pipeline_params.input_data_path)
    info = data.copy()

    isochrone_30 = pd.read_csv(train_pipeline_params.input_isochrone_30_path)
    isochrone_25 = pd.read_csv(train_pipeline_params.input_isochrone_25_path)
    isochrone_20 = pd.read_csv(train_pipeline_params.input_isochrone_20_path)
    isochrone_15 = pd.read_csv(train_pipeline_params.input_isochrone_15_path)
    isochrone_10 = pd.read_csv(train_pipeline_params.input_isochrone_10_path)
    distance = pd.read_csv(train_pipeline_params.input_distance_path)

    highway = pd.read_csv(train_pipeline_params.input_highway_path)
    highway.drop(["name_highway", "Номерточки"], axis=1, inplace=True)

    reestr = pd.read_csv(train_pipeline_params.input_reestr_path)
    reestr.drop(["Номерточки"], axis=1, inplace=True)

    get_difference(isochrone_30, isochrone_25)
    get_difference(isochrone_25, isochrone_20)
    get_difference(isochrone_20, isochrone_15)
    get_difference(isochrone_15, isochrone_10)

    work_schedule = pd.DataFrame(
        WORK_SCHEDULE,
        columns=[
            "График",
            "Будни начало",
            "Будни конец",
            "Выходные начало",
            "Выходные конец",
            "Разные графики",
        ],
    )
    data = pd.merge(data, work_schedule, on="График", how="left")
    data["Рабочие часы в будни"] = data["Будни конец"] - data["Будни начало"]
    data["Рабочие часы в выходные"] = data["Выходные конец"] - data["Выходные начало"]
    data["Ночной магазин"].replace({"Нет": 0, "Да": 1}, inplace=True)

    data = pd.concat([data, add_isochrone(isochrone_10, "10")], axis=1)
    data = pd.concat([data, add_isochrone(isochrone_15, "15")], axis=1)
    data = pd.concat([data, add_isochrone(isochrone_20, "20")], axis=1)
    data = pd.concat([data, add_isochrone(isochrone_25, "25")], axis=1)
    data = pd.concat([data, add_isochrone(isochrone_30, "30")], axis=1)
    data = pd.concat([data, distance], axis=1)
    data = pd.concat([data, highway], axis=1)
    data = pd.concat([data, reestr], axis=1)

    data_categorical = data[["Регион", "Город"]]
    data = data.drop(
        columns=[
            "Дата открытия",
            "Наименование",
            "Номерточки",
            "Регион",
            "Город",
            "Адрес",
            "Широта",
            "Долгота",
            "График",
        ],
        axis=1,
    )
    top_20 = [
        "Торговая площадь, м2",
        "Будни конец",
        "Ночной магазин",
        "Выходные конец",
        "Рабочие часы в будни",
        "Рабочие часы в выходные", 
        "Будни начало",
        "Выходные начало",
        "stations15",
        "area_common_property",
        "medicine10",
        "food20",
        "house_count",
        "unliving_quarters_count",
        "area_non_residential",
        "stations10",
        "stations20",
        "parking_square",
        "shops30",
        "atms+banks20",
    ]

    return info, data[top_20], target


def get_difference(largest_iso, least_iso):
    for col in [
        "bus_stop",
        "house",
        "kiosk",
        "retail",
        "station",
        "subway_entrance",
        "tram_stop",
        "bar",
        "cafe",
        "fast_food",
        "food_court",
        "pub",
        "restaurant",
        "college",
        "driving_school",
        "language_school",
        "school",
        "kindergarten",
        "university",
        "car_wash",
        "fuel",
        "atm",
        "bank",
        "clinic",
        "dentist",
        "doctors",
        "hospital",
        "pharmacy",
        "veterinary",
        "theatre",
        "cinema",
        "hostel",
        "hotel",
        "office",
        "shop",
    ]:
        largest_iso[col] = largest_iso[col] - least_iso[col]


def add_isochrone(new_features, isochrone):
    new_features_aggregated = pd.DataFrame(
        columns=[
            "medicine" + isochrone,
            "stations" + isochrone,
            "housing" + isochrone,
            "shops" + isochrone,
            "atms+banks" + isochrone,
            "office" + isochrone,
            "food" + isochrone,
            "for_motorists" + isochrone,
        ]
    )

    new_features_aggregated["medicine" + isochrone] = (
        new_features["clinic"]
        + new_features["dentist"]
        + new_features["doctors"]
        + new_features["hospital"]
        + new_features["pharmacy"]
    )

    new_features_aggregated["food" + isochrone] = (
        new_features["food_court"]
        + new_features["pub"]
        + new_features["restaurant"]
        + new_features["cafe"]
        + new_features["bar"]
        + new_features["fast_food"]
    )

    new_features_aggregated["stations" + isochrone] = (
        new_features["bus_stop"]
        + new_features["station"]
        + new_features["subway_entrance"]
        + new_features["tram_stop"]
    )

    new_features_aggregated["housing" + isochrone] = (
        new_features["hotel"] + new_features["hostel"]
    )

    new_features_aggregated["for_motorists" + isochrone] = (
        new_features["car_wash"] + new_features["fuel"]
    )

    new_features_aggregated["shops" + isochrone] = (
        new_features["kiosk"] + new_features["retail"] + new_features["shop"]
    )

    new_features_aggregated["atms+banks" + isochrone] = (
        new_features["atm"] + new_features["bank"]
    )

    new_features_aggregated["office" + isochrone] = new_features["office"]

    return new_features_aggregated
