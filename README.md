# Проект №15. Процедура аудита ошибок и объяснений классификатора изображений

[![Открыть в Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ampldariik/xai-audit-oxford-pets/blob/main/notebooks/project15_demo.ipynb)

**Демо в Google Colab:** ноутбук `notebooks/project15_demo.ipynb` сам скачивает датасет, показывает данные, сплит, метрики, Grad-CAM, подмену фона и все результаты. Обучать заново не нужно — модели лежат в `runs/`.


**Дисциплина:** Аналитика больших массивов (В22.045) · Yessenov University · 4 курс, 7 семестр, 2026–2027

**Вопрос исследования:** на что опирается классификатор пород кошек и собак: на само животное или
на фон и посторонние артефакты (водяные знаки)? Какие у него типичные ошибки и почему?

**Датасет:** Oxford-IIIT Pet: 7 349 фото, 37 пород, у каждого кадра есть маска животного (trimap).
Лицензия CC BY-SA 4.0. Источник: <https://www.robots.ox.ac.uk/~vgg/data/pets/>.

## Быстрый старт

```bash
pip install -r requirements.txt
mkdir -p data/raw   # положите сюда images.tar.gz и annotations.tar.gz (или .tar)
bash run_all.sh     # весь конвейер; результаты в reports/
```

Код сам выбирает устройство (CUDA → Apple MPS → CPU). На Google Colab с GPU T4 весь прогон
занимает около 30 минут, на двух ядрах CPU около 5 часов (из них ~1,5 ч — три варианта водяного знака). Все параметры задаются в
`configs/default.yaml`, seed фиксирован (42).

## Результаты (прогон 09.10.2026, CPU, 2 потока)

| Что | Результат | Файл |
|---|---|---|
| Сплит | 53 пары почти-дубликатов; при случайном сплите 15 пересекали бы train/test, у нас 0 | `reports/data_passport.json` |
| Baseline-0 (LogReg, цвет) | Macro-F1 0,091 | `reports/model_comparison.csv` |
| Baseline-1 (CNN с нуля) | Macro-F1 0,446 | |
| Linear probe (LogReg на признаках ImageNet) | Macro-F1 0,889 | |
| **ResNet-18 + fine-tuning** | **Macro-F1 0,894 [0,875; 0,911]**, accuracy 0,895 | `reports/metrics_main.json` |
| Grad-CAM | 82 % карты на животном, pointing game 98,8 % | `reports/xai_summary.json` |
| Контрольное вмешательство | удаление топ-20 % Grad-CAM: −0,64 p, случайных 20 %: −0,12; рандомизация: ρ = 0,06 | `reports/saliency_check.json` |
| Подмена фона | смена ответа: ResNet-18 8–11 %, CNN с нуля 44–56 % | `reports/bg_swap_metrics.csv` |
| Водяной знак (после аугментаций) | recall целевого класса 0,45 → 0,79 с надписью; Grad-CAM в зоне надписи 1 % → 17 % | `reports/watermark_results_after_aug.json` |
| Атлас 110 ошибок | похожие породы 50 %, фон 23 %, качество 16 %, прочее 11 % | `reports/error_atlas.md` |
| Масштаб | N = 1000: 74 с (13,4 кадра/с, батч 16); время ×8,7–9,6 при N ×10; RAM зависит от батча | `reports/benchmark_summary.json` |

## Структура

```
configs/default.yaml        все гиперпараметры и пути
src/
  data.py                   разметка, кэш (memmap), признаки качества, pHash-группы, сплит без утечек
  models.py                 BaselineCNN (с нуля), ResNet-18/50 (ImageNet) + fine-tuning
  train.py                  цикл обучения, выбор эпохи по val Macro-F1
  metrics.py                Accuracy, Macro-F1/P/R, бутстреп-ДИ, матрица ошибок
  xai.py                    Grad-CAM, Integrated Gradients, FG-energy / pointing game
  background.py             вырезание объекта, inpainting фона, 6 вариантов подмены фона
  watermark.py              подсаженный водяной знак (положительный контроль)
scripts/
  01_prepare_data.py        неделя 2:  паспорт данных, EDA, почти-дубликаты, сплит
  02_train.py               неделя 5:  color_logreg | baseline_cnn | main
  02b_linear_probe.py       неделя 5:  scikit-learn LogReg на замороженных признаках ImageNet
  03_evaluate.py            неделя 5:  метрики на тесте, сравнение моделей
  04_xai.py                 неделя 10: карты значимости и их количественная оценка
  04b_saliency_check.py     неделя 10: проверка карт контрольным вмешательством (удаление + рандомизация)
  05_background_swap.py     неделя 10: эксперимент с подменой фона
  06_watermark.py           неделя 10: эксперимент с водяным знаком
  07_error_atlas.py         неделя 14: атлас ошибок
  08_benchmark.py           неделя 14: время и память на N=100 и N=1000
reports/                    JSON/CSV с результатами + figures/*.png для отчёта
```

## Соответствие чек-листу курса

| Требование | Где реализовано | Ключевой артефакт |
|---|---|---|
| Паспорт задачи и данных | `01_prepare_data.py` | `reports/data_passport.json` |
| Сплит без утечек | `src/data.py` (pHash + StratifiedGroupKFold) | `near_duplicate_pairs.csv`, раздел near_duplicates в паспорте |
| Baseline vs основная модель (+ sklearn linear probe) | `02_train.py`, `02b_linear_probe.py`, `03_evaluate.py` | `model_comparison.csv`, `figures/confusion_*.png` |
| Macro-F1, Accuracy, матрица ошибок | `src/metrics.py` | `metrics_*.json` (с 95% ДИ) |
| XAI (Grad-CAM, IG) | `04_xai.py` | `xai_summary.json`, `figures/xai_*.png` |
| Проверка карты контрольным вмешательством | `04b_saliency_check.py` | `saliency_check.json`, `figures/saliency_deletion.png` |
| Подмена фона, Δ Confidence | `05_background_swap.py` | `bg_swap_metrics.csv`, `figures/bg_swap_*.png` |
| Водяные знаки | `06_watermark.py` | `watermark_results.json`, `figures/watermark_gradcam.png` |
| Атлас ошибок | `07_error_atlas.py` | `error_atlas.md`, `error_atlas.csv`, `figures/atlas_*.png` |
| Производительность N=100 / 1000 | `08_benchmark.py` | `benchmark_summary.json`, `figures/benchmark.png` |

## Как устроена защита от утечек

1. **Почти-дубликаты.** Для каждого кадра считается перцептивный хэш pHash (64 бита). Кадры с
   расстоянием Хэмминга ≤ 8 объединяются в одну «серию» через Union-Find. Деление 70/15/15 делается
   через `StratifiedGroupKFold`: серия целиком попадает в одну подвыборку, пропорции пород сохраняются.
2. Аугментации применяются **только к train**. Нормализация использует фиксированные статистики ImageNet,
   поэтому статистика по test не считается.
3. Эпоха модели выбирается **только по validation**. Тест открывается один раз, в `03_evaluate.py`.
4. Пороги «низкого качества» в атласе считаются **только по train**.
5. Фоны для подмены берутся из **validation**, а не из теста.

## Если download.pytorch.org недоступен

`src/models.py` сам переключается на зеркало весов ImageNet на GitHub Releases
(timm, ResNet strikes back, `resnet18_a1_0`). Архитектура и имена слоёв совпадают с torchvision.
