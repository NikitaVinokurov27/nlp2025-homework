# nlp2025-homework

Домашние работы по курсу [tam2511/nlp2025](https://github.com/tam2511/nlp2025).

## Структура

```
lesson2/homeworks/agnews_homework.ipynb   — классификация AG News (цель: Macro F1 ≥ 0.95)
```

## lesson2 — Классификация AG News

**Задание**: достичь Macro F1 ≥ 0.95 на тесте AG News (одна модель, без ансамблей).

**Подход**: гибрид CNN + BiLSTM + self-attention с предобученными Word2Vec эмбеддингами.

**Архитектура**:
- Embedding — pretrained `word2vec-google-news-300` (trainable, padding_idx=0)
- Spatial dropout по каналам эмбеддингов + word dropout
- Параллельные Conv1d с ядрами {2, 3, 4, 5}, по 128 фильтров → ReLU → masked max-over-time pooling
- BiLSTM(hidden=128, 1 слой, bidirectional) c `pack_padded_sequence` → additive self-attention pooling по маске
- Concat(CNN, attn) → BatchNorm → Dropout → Linear(256) → ReLU → Dropout → Linear(4)

**Обучение**:
- AdamW (lr=1e-3, weight_decay=1e-5) + CosineAnnealingLR
- CrossEntropyLoss с label_smoothing=0.1
- Gradient clipping = 1.0
- ModelCheckpoint по `val_f1`, EarlyStopping(patience=4)
- 12 эпох, batch_size=128, train/val 90/10 (стратифицированно)

**Отчёт в ноутбуке**:
- Графики train/val loss, accuracy, Macro F1 по эпохам
- Classification report
- Confusion matrix (raw + normalized)
- Анализ ошибок (10 примеров)

## Как запустить

```bash
# зависимости
pip install torch pytorch-lightning torchmetrics datasets gensim scikit-learn matplotlib seaborn

# запустить ноутбук
jupyter notebook lesson2/homeworks/agnews_homework.ipynb
```

В ячейке `## Шаг 0: Конфигурация` есть флаг `SMOKE_TEST`:
- `SMOKE_TEST = True` — быстрый прогон на 2000 примерах / 1 эпоха (для проверки на CPU)
- `SMOKE_TEST = False` — полное обучение (нужен GPU; ~10–15 мин на эпоху на T4/L40)

При желании можно заменить `EMBEDDING_NAME = 'word2vec-google-news-300'` (1.6 GB) на
`'glove-wiki-gigaword-300'` (~370 MB) или `'fasttext-wiki-news-subwords-300'`.

## Использованные материалы лекции

Семинарские ноутбуки [tam2511/nlp2025/lesson2/seminar](https://github.com/tam2511/nlp2025/tree/main/lesson2/seminar):

- `bilstm_agnews.ipynb` — структура Lightning-модуля, masked max-pooling
- `textcnn_agnews.ipynb` — TextCNN, токенизация и vocab
- `bilstm_word2vec.ipynb` — загрузка предобученных Word2Vec эмбеддингов
