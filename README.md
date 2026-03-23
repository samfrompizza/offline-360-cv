# Drone Panorama MVP (classic branch)

Минимальный офлайн-пайплайн для поиска маленьких ярких дронов на панорамном видео.

## Что делает проект теперь

- читает видео `.mov` / `.mp4`;
- использует только **classical**-детекцию без YOLO;
- ищет небольшие движущиеся яркие/насыщенные объекты на фоне неподвижных ворот;
- строит простой фон через exponential moving average и умеет **быстрее забывать прошлые кадры**;
- выполняет простое межкадровое отслеживание объектов;
- умеет опционально включать **простой cone filter** по ожидаемой позиции дрона;
- помечает почти неподвижные объекты как `static`;
- отбрасывает статичные треки, чтобы уменьшить ложные срабатывания на воротах;
- сохраняет:
  - annotated video;
  - JSON с `track_id` по кадрам.

## Почему это подходит для панорамного видео с дронами

В таких видео дроны очень маленькие, быстрые и контрастные, а цветные ворота статичны. Поэтому текущий пайплайн использует три идеи:

1. **Classical motion + color baseline** ищет именно небольшие движущиеся яркие объекты.
2. **Background EMA** сравнивает кадр с фоном и обновляет фон с настраиваемой скоростью. Это помогает уменьшить шлейф после пролёта дрона.
3. **Static filter + optional cone filter** дополнительно уменьшают ложные совпадения на неподвижных воротах и при пересечениях траекторий.

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

Ускорить забывание старых кадров:

```bash
python scripts/run_inference.py --video data/raw/1.mov --background-alpha 0.5
```

Принудительно выключить cone filter, даже если он включён в конфиге:

```bash
python scripts/run_inference.py --video data/raw/1.mov --disable-cone-filter
```

Результаты будут сохранены в `outputs/`:

- `*_annotated.mp4` — видео с bbox и `track_id`;
- `*_tracks.json` — JSON по кадрам.

## Конфиг inference

Параметры лежат в `configs/inference.yaml`.

Ключевые настройки:

- `classical.background_alpha`: скорость обновления фона. Чем больше значение, тем быстрее алгоритм забывает старые кадры и тем меньше шлейф;
- `classical.morph_kernel_size`: размер морфологического ядра;
- `classical.dilate_iterations`: дополнительное расширение motion mask; по умолчанию выключено, чтобы не раздувать след;
- `tracker.max_missed_frames`: сколько кадров трек может прожить без подтверждения;
- `tracker.use_cone_filter`: включает простой фильтр ожидаемой позиции;
- `tracker.cone_*`: параметры конуса ожидаемой позиции.

## Структура

- `scripts/run_inference.py` — основной CLI для классической детекции, трекинга и сохранения результатов.
- `src/detector.py` — classical detector и NMS.
- `src/tracker.py` — простой трекер по IoU + расстоянию центра с опциональным cone filter.
- `src/postprocess.py` — фильтр полезных движущихся треков.
- `src/visualize.py` — отрисовка и сохранение JSON.
- `src/video_io.py` — чтение/запись видео.
- `scripts/extract_frames.py` — сбор кадров для ручного анализа.

## Что было удалено из ветки classic

Из репозитория убраны:

- YOLO inference;
- tile-based разбиение кадра;
- train/export-заготовки под YOLO;
- зависимость `ultralytics`.

## Ограничения MVP

- Классический метод всё ещё чувствителен к motion blur и сильным изменениям освещения.
- При большом количестве дронов одновременно простой cone filter может иногда ошибаться, но он намеренно оставлен очень лёгким и опциональным.
- Трекер по-прежнему MVP-уровня и не претендует на research-grade качество.
