# Super-Resolution: SRCNN + сравнение с ESRGAN

Увеличение разрешения изображений в 2 и 4 раза с восстановлением деталей.
Основная модель — **SRCNN** (обучается в проекте), для сравнения — **ESRGAN** и **Real-ESRGAN**
(официальные предобученные веса, скачиваются автоматически) и бикубическая интерполяция.

Метрики: **PSNR, SSIM** (канал Y, как в статьях), **LPIPS**, анализ ошибок деталей
(ошибка I рода — «выдуманные» детали, ошибка II рода — потерянные), **MOS** (субъективная оценка).

## Структура

```
sr_project/
├── upscale.py            # ПРИЛОЖЕНИЕ: файл/папка/веб-камера → окно с результатом + log.txt
├── train.py              # обучение SRCNN (×2 / ×4), валидация PSNR/SSIM, графики
├── evaluate.py           # тест: PSNR/SSIM/LPIPS/ошибки, таблица summary.md, иллюстрации
├── prepare_data.py       # разбиение train/val/test + нарезка обучающих картинок
├── download_datasets.py  # DIV2K, Flickr2K, Set5/Set14/BSD100/Urban100
├── mos.py                # субъективная оценка MOS (слепой тест, оценки 1–5)
├── make_report_figures.py# графики для отчёта из логов обучения и результатов теста
├── srlib/                # модели, метрики, обработка изображений, датасет
├── checkpoints/          # веса SRCNN (srcnn_x2.pth, srcnn_x4.pth)
├── weights/              # веса ESRGAN / Real-ESRGAN (скачиваются сами)
├── runs/                 # логи и графики обучения
└── results/              # результаты: апскейлы, метрики, рисунки
```

## 1. Установка (Windows / Linux, RTX 3050 Ti)

```bash
python -m venv venv
venv\Scripts\activate            # Linux: source venv/bin/activate
# PyTorch с поддержкой CUDA (команду под свою версию драйвера можно взять на pytorch.org):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Должно вывести `True NVIDIA GeForce RTX 3050 Ti Laptop GPU`.

## 2. Быстрый старт (без обучения)

В `checkpoints/` уже лежат веса SRCNN из контрольного прогона — приложение работает сразу:

```bash
python upscale.py --input photo.jpg --model srcnn --scale 4
python upscale.py --input photo.jpg --model esrgan --scale 4 --mode eval
python upscale.py --webcam 0 --model srcnn --scale 2
```

## 3. Данные и разбиение

```bash
python download_datasets.py div2k benchmarks
python prepare_data.py --train-dirs datasets/DIV2K_train_HR --val-dir datasets/DIV2K_valid_HR --val-count 10 --test-dirs Set5=datasets/Set5 Set14=datasets/Set14 BSD100=datasets/BSD100 Urban100=datasets/Urban100
```

| Часть | Что | Зачем |
|---|---|---|
| train | DIV2K_train_HR, 800 изобр. (+ по желанию Flickr2K: `--train-dirs datasets/DIV2K_train_HR datasets/Flickr2K/Flickr2K_HR`) | обучение |
| val | первые 10 из DIV2K_valid_HR | выбор лучшего чекпоинта по PSNR |
| test | остальные 90 из DIV2K_valid (`DIV2K_valid_HR_rest`) + Set5, Set14, BSD100, Urban100 | итоговая оценка, в обучении не участвуют |

LR-изображения не хранятся: они получаются из HR бикубическим уменьшением «на лету» (стандарт DIV2K track 1).
Обучающие картинки нарезаются на фрагменты 192×192 (канал Y) в `data/train_y.npy` (~2 ГБ для DIV2K) —
это убирает декодирование 2K-PNG из цикла обучения.

## 4. Обучение SRCNN

```bash
python train.py --scale 2        # ~10–15 мин на RTX 3050 Ti
python train.py --scale 4
```
По умолчанию: 20 000 итераций, batch 64, патч 96×96, Adam (lr 1e-3 для слоёв 1–2 и 1e-4 для последнего),
косинусное снижение lr, MSE loss, mixed precision. Валидация каждые 1000 итераций.

Полезные флаги: `--fast` (3000 итераций для проверки), `--resume` (продолжить), `--residual`
(глобальная skip-связь — быстрее сходится), `--batch 32` (если не хватает памяти).

Выход: `checkpoints/srcnn_x{S}.pth` (лучший по PSNR на val), `runs/srcnn_x{S}/train_log.csv`, `curves.png`.

## 5. Оценка на тесте

```bash
python evaluate.py --lpips
python make_report_figures.py
```
`results/summary.md` — таблица средних PSNR/SSIM/LPIPS/ошибок по наборам;
`results/figures/` — успешные кейсы, ошибки I/II рода, графики для отчёта.

## 6. Приложение `upscale.py`

| Параметр | Значение |
|---|---|
| `--input PATH` | изображение или папка |
| `--webcam [N]` | веб-камера в реальном времени |
| `--model` | `srcnn` (по умолч.), `esrgan`, `realesrgan`, `bicubic` |
| `--scale` | `2` или `4` |
| `--mode sr` | обычный апскейл (для файлов по умолчанию); эталона нет → в лог пишется LR-PSNR |
| `--mode eval` | вход считается эталоном: ↓S → модель → PSNR/SSIM относительно оригинала (для камеры по умолчанию) |
| `--ref FILE` | эталон для подсчёта PSNR/SSIM в режиме `sr` |
| `--lpips` | дополнительно LPIPS |
| `--no-show` | без окна |

Окно: слева бикубическая интерполяция, справа модель, снизу — увеличенные фрагменты под курсором
(вход LR, бикубическая, модель, эталон). Клавиши: `s` — снимок окна, `q`/`Esc` — выход, пробел — пауза (камера).

Результат сохраняется в `results/upscaled/`, каждая обработка дописывается в `log.txt`:
```
2026-09-26 22:04:13 | input=datasets/Set14/img_005.png | mode=eval model=srcnn scale=2 device=cuda tile=1024 fp16=True checkpoint=srcnn_x2.pth | in_size=125x180 out_size=250x360 time=0.004s PSNR=27.47dB SSIM=0.8930 PSNR_bicubic=26.02dB SSIM_bicubic=0.8496 | output=results/upscaled/img_005_srcnn_x2_eval.png
```

## 7. MOS (субъективная оценка)

```bash
python mos.py prepare --images datasets/Set14 --scale 4 --n 10
python mos.py rate --rater ivan        # каждый оценщик; клавиши 1–5
python mos.py summary
```
