# Drone Panorama MVP

Минимальный офлайн-пайплайн для поиска дронов на панорамном видео.

## Что делает проект теперь

- читает видео `.mov` / `.mp4`;
- поддерживает **2 способа детекции**:
  - **classical** — через motion + color/contrast heuristics для маленьких быстрых объектов;
  - **yolo** — baseline inference через `ultralytics` + `yolo11n.pt` по перекрывающимся tiles;
- умеет запускать оба способа сразу через `method: both`;
- для YOLO:
  - автоматически подгружает `yolo11n.pt`;
  - режет кадр на перекрывающиеся tiles;
  - делает inference по каждому tile;
  - переносит bbox обратно в координаты исходного кадра;
  - объединяет пересекающиеся детекции через NMS;
- выполняет простое межкадровое отслеживание объектов;
- помечает почти неподвижные объекты как `static`;
- отбрасывает статичные треки, чтобы уменьшить ложные срабатывания на воротах;
- сохраняет:
  - annotated video;
  - JSON с `track_id` по кадрам.

## Почему это подходит для панорамного видео с дронами

В таких видео дроны очень маленькие, быстрые и контрастные, а цветные ворота статичны. Поэтому MVP использует две идеи:

1. **Classical baseline** сначала ищет именно движение и яркие/насыщенные мелкие объекты. Это помогает отделить движущийся дрон от неподвижных ворот.
2. **YOLO baseline** режет широкий кадр на tiles, чтобы маленький объект занимал больше пикселей на входе модели.
3. **Static filter** дополнительно отбрасывает треки, у которых bbox и центр почти не меняются много кадров подряд.

## Установка

### WSL / Ubuntu
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Запуск inference

Базовый запуск:

```bash
python scripts/run_inference.py --video data/input.mp4
```

Только классический метод:

```bash
python scripts/run_inference.py --video data/input.mp4 --method classical
```

Только YOLO tile inference:

```bash
python scripts/run_inference.py --video data/input.mp4 --method yolo
```

Результаты будут сохранены в `outputs/`:

- `*_annotated.mp4` — видео с bbox и `track_id`;
- `*_tracks.json` — JSON по кадрам.

## Конфиг inference

Параметры лежат в `configs/inference.yaml`.

Ключевые настройки:

- `method`: `classical`, `yolo` или `both`;
- `classical.*`: пороги для motion/color baseline;
- `yolo.model_path`: должен быть `yolo11n.pt`;
- `yolo.tile_size`, `yolo.tile_overlap`: размер tile и перекрытие;
- `tracker.*`: параметры простого MVP-трекера;
- `min_track_hits`: сколько подтверждений нужно до вывода трека.

## Структура

- `scripts/run_inference.py` — основной CLI для детекции, трекинга и сохранения результатов.
- `src/tiling.py` — разбиение кадра на tiles с перекрытием.
- `src/detector.py` — classical detector и YOLO tile detector.
- `src/tracker.py` — простой трекер по IoU + расстоянию центра.
- `src/postprocess.py` — фильтр полезных движущихся треков.
- `src/visualize.py` — отрисовка и сохранение JSON.
- `src/video_io.py` — чтение/запись видео.
- `scripts/prepare_dataset.py` — сбор кадров в датасет для разметки.
- `scripts/train.py` — обучение своих весов.
- `scripts/export_onnx.py` — экспорт в ONNX для будущего Kotlin/live.

## Ограничения MVP

- `yolo11n.pt` здесь используется как **baseline**, без дообучения на дронах; качество на очень маленьких объектах может быть ограничено.
- Классический метод опирается на движение, поэтому при сильном motion blur или резких сдвигах камеры потребуются подстройки порогов.
- Трекер намеренно простой и не претендует на research-grade качество.
