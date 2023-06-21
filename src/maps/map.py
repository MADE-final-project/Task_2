import os
import shutil
import base64
import pandas as pd
import shap
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster, HeatMap


def plotting_map(train_pipeline_params, model, features, result) -> None:
    os.mkdir("fig")
    explainer = shap.Explainer(model)
    shap_values = explainer(features)

    for idx, row in result.iterrows():
        plt.clf()
        png = f'fig/{row["Номерточки"]}.png'
        shap.plots.waterfall(shap_values[idx], max_display=80, show=False)
        plt.savefig(png, dpi=100, bbox_inches="tight")

    resolution, width, height = 100, 12, 5
    m = folium.Map(
        location=[result["Широта"].mean(), result["Долгота"].mean()], zoom_start=10
    )
    marker_cluster = MarkerCluster().add_to(m)
    heat_data = []
    for idx, row in result.iterrows():
        location = [row["Широта"], row["Долгота"]]
        heat_data.append([row["Широта"], row["Долгота"], row["Предсказание выручки"]])

        png = f'fig/{row["Номерточки"]}.png'
        encoded = base64.b64encode(open(png, "rb").read())
        html = '<img src="data:image/png;base64,{}">'.format
        iframe = folium.IFrame(
            html(encoded.decode("UTF-8")),
            width=(width * resolution) + 20,
            height=(height * resolution) + 20,
        )
        popup = folium.Popup(iframe, max_width=2650)

        folium.Marker(
            location,
            popup=popup,
            tooltip=f"""
            <b>{row["Широта"]}</b>, <b>{row["Долгота"]}</b><br>
            <i>Номерточки: </i><b><br>{row['Номерточки']}</b><br>
            <i>Чеки: </i><b><br>{round(row['Чеки шт/мес'], 2)}</b><br>
            <i>Выручка: </i><b><br>{round(row['Выручка р/мес'], 2)}</b><br>
            <i>Предсказание выручки: </i><b><br>{round(row['Предсказание выручки'], 2)}</b><br>""",
        ).add_to(marker_cluster)

    HeatMap(
        heat_data, radius=18, gradient={0.4: "blue", 0.65: "lime", 1: "red"}
    ).add_to(m)
    colormap = folium.branca.colormap.LinearColormap(
        ["blue", "lime", "red"],
        vmin=result["Предсказание выручки"].min(),
        vmax=result["Предсказание выручки"].max(),
        caption="Предсказание выручки",
    )
    colormap.add_to(m)
    m.save(train_pipeline_params.output_map_path)

    shutil.rmtree("fig")
