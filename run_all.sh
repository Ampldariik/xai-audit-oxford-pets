#!/usr/bin/env bash
# Полный воспроизводимый прогон проекта. Перед запуском положите
# images.tar.gz и annotations.tar.gz (или .tar) в data/raw/.
# Время: GPU (Colab T4) ≈ 15–20 мин; CPU (2 ядра) ≈ 3–4 ч.
set -euo pipefail
cd "$(dirname "$0")"
CFG=${1:-configs/default.yaml}

python scripts/01_prepare_data.py   "$CFG"   # неделя 2: паспорт, EDA, pHash, сплит
python scripts/02_train.py color_logreg "$CFG" # неделя 5: baseline-0
python scripts/02_train.py baseline_cnn "$CFG" # неделя 5: baseline-1
python scripts/02_train.py main         "$CFG" # неделя 5: ResNet-18 fine-tuning
python scripts/02b_linear_probe.py "$CFG"      # неделя 5: sklearn LogReg на замороженных признаках ImageNet
python scripts/03_evaluate.py       "$CFG"   # неделя 5: метрики, матрицы ошибок
python scripts/04_xai.py            "$CFG"   # неделя 10: Grad-CAM, IG
python scripts/04b_saliency_check.py "$CFG"  # неделя 10: контрольное вмешательство (deletion + randomization)
python scripts/05_background_swap.py "$CFG"  # неделя 10: подмена фона
python scripts/06_watermark.py      "$CFG"   # неделя 10: водяной знак 80 %, реалистичный (кроп может срезать надпись)
python scripts/06_watermark.py configs/watermark_strong.yaml     # 100 %, непрозрачность 0.9
python scripts/06_watermark.py configs/watermark_after_aug.yaml  # положительный контроль: надпись после аугментаций
python scripts/07_error_atlas.py    "$CFG"   # неделя 14: атлас ошибок
python scripts/08_benchmark.py      "$CFG"   # неделя 14: N=100 vs N=1000, время и память
echo "Все результаты в reports/"
