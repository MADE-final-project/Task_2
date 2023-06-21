# Task_2  
**Прогнозирование выручки на основе ограниченного числа наблюдений**  

-----  

1. Установить зависимости:
```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Создать в корне проекта папку *data*, скачать и разместить в ней [данные](https://drive.google.com/drive/folders/1GmMMIrYE3U-dgqFBKEpVEjLEFDQEucbL?usp=drive_link).

3. Запустить обучение модели:
```sh
python src/train_pipeline.py configs/train_config.yaml
```
*Примечание:* указать в конфиге флаг plotting_map=True, если необходимо отобразить результат на карте.