# Drone Panorama MVP

Минимальный офлайн-пайплайн для поиска дронов на панорамном видео.

## Что делает проект сейчас

- читает видео `.mov` / `.mp4`
- извлекает кадры с заданным шагом
- подготавливает базу для дальнейшей детекции объектов
- позже сюда добавятся:
  - tiling / cubemap
  - YOLO inference
  - трекинг
  - фильтр статичных объектов
  - обучение на своих данных

## Установка

### WSL / Ubuntu
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

extract_frames.py — вытаскивает кадры из видео.

tiling.py — режет кадр на окна с перекрытием.

detector.py — запускает YOLO на каждом tile.

tracker.py — связывает детекции между кадрами.

postprocess.py — отбрасывает неподвижные объекты.

visualize.py — рисует bbox и сохраняет итоговое видео.

prepare_dataset.py — собирает кадры в датасет для разметки.

train.py — обучение на своих данных.

export_onnx.py — экспорт в ONNX для будущего Kotlin/live.