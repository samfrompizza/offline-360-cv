# Drone Panorama MVP

Минимальная офлайн-утилита для поиска красных объектов на панорамном видео.

## Что делает проект

- читает видео `.mov` / `.mp4`;
- использует только **classical**-детекцию на основе классических алгоритмов без нейронных сетей;
- ищет любые объекты в захардкоженном красном HSV-диапазоне, например красное яблоко;
- не требует движения объекта: статичные красные объекты тоже попадают в результат;
- выполняет простое межкадровое отслеживание объектов;
- умеет опционально включать **простой cone filter** по ожидаемой позиции объекта;
- сохраняет:
  - annotated video с bbox без лишних подписей;
  - компактный JSON с координатами найденных объектов по кадрам.

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
python scripts/run_inference.py --video data/raw/1.mov
```

Включить простой cone filter:

```bash
python scripts/run_inference.py --video data/raw/1.mov --use-cone-filter
```

Принудительно выключить cone filter, даже если он включён в конфиге:

```bash
python scripts/run_inference.py --video data/raw/1.mov --disable-cone-filter
```

Результаты будут сохранены в `outputs/`:

- `*_annotated.mp4` — видео с bbox без текста;
- `*_tracks.json` — JSON по кадрам с `bbox_xyxy`, `center_xy` и нормализованным `center_norm` для Android-оверлея.

## Конфиг inference

Параметры лежат в `configs/inference.yaml`.

Ключевые настройки:

- `classical.color_threshold`: минимальные S/V компоненты внутри красного HSV-диапазона;
- `classical.morph_kernel_size`: размер морфологического ядра;
- `classical.dilate_iterations`: дополнительное расширение красной маски; по умолчанию выключено, чтобы не раздувать bbox;
- `min_track_hits`: сколько раз трек должен быть подтверждён перед записью в overlay metadata;
- `tracker.max_missed_frames`: сколько кадров трек может прожить без подтверждения;
- `tracker.use_cone_filter`: включает простой фильтр ожидаемой позиции;
- `tracker.cone_*`: параметры конуса ожидаемой позиции.

## Структура

- `scripts/run_inference.py` — основной CLI для классической детекции, трекинга и сохранения результатов.
- `src/detector.py` — classical detector.
- `src/tracker.py` — простой трекер по IoU + расстоянию центра с опциональным cone filter.
- `src/postprocess.py` — фильтр подтверждённых красных треков.
- `src/visualize.py` — отрисовка и сохранение JSON.
- `src/video_io.py` — чтение/запись видео.
- `scripts/extract_frames.py` — сбор кадров для ручного анализа.

## Ограничения MVP

- Классический метод чувствителен к сильным изменениям освещения и оттенкам красного вне заданного HSV-диапазона.
- При большом количестве объектов одновременно простой cone filter может иногда ошибаться, но он намеренно оставлен очень лёгким и опциональным.
