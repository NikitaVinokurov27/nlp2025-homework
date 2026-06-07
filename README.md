# nlp2025-homework

Домашка по курсу [tam2511/nlp2025](https://github.com/tam2511/nlp2025).

## lesson2 — Классификация AG News

Цель — Macro F1 ≥ 0.95 на тесте.

Сделал два ноутбука, чтобы было видно прогрессию от baseline к улучшенной модели:

| Ноутбук | Модель | Эпох | Test Macro F1 | Test Accuracy |
|---|---|---|---|---|
| [`agnews_textcnn.ipynb`](lesson2/homeworks/agnews_textcnn.ipynb) | TextCNN (kernels=2,3,4,5) + GloVe-100 | 5 | **0.9262** | 0.9263 |
| [`agnews_hybrid.ipynb`](lesson2/homeworks/agnews_hybrid.ipynb) | TextCNN + BiLSTM(hidden=128) + GloVe-100 | 10 | **0.9333** | 0.9334 |

Оба ноутбука прогонялись локально на CPU (без GPU), все ячейки выполнены, в файлах уже встроены выводы — графики обучения, classification report, confusion matrix и разбор ошибок. Можно открыть прямо на GitHub и посмотреть.

### Что взял из семинара

- Структура TextCNN (kernels 3/4/5) — из [`textcnn_agnews.ipynb`](https://github.com/tam2511/nlp2025/blob/main/lesson2/seminar/textcnn_agnews.ipynb)
- Masked max-pool для BiLSTM — из [`bilstm_agnews.ipynb`](https://github.com/tam2511/nlp2025/blob/main/lesson2/seminar/bilstm_agnews.ipynb)
- Загрузка предобученных эмбеддингов через `gensim.downloader` — из [`bilstm_word2vec.ipynb`](https://github.com/tam2511/nlp2025/blob/main/lesson2/seminar/bilstm_word2vec.ipynb)

### Что добавил поверх семинара

- Kernel=2 в CNN (немного помог на коротких заголовках)
- BiLSTM поверх тех же эмбеддингов, склейка с CNN-фичами в общий MLP
- AdamW + CosineAnnealingLR (вместо просто Adam)
- Label smoothing 0.1
- BatchNorm перед классификатором
- Gradient clipping 1.0
- Токен `<num>` для чисел — улучшил покрытие словаря на финансовых заголовках

### Честно о результате

На CPU за разумное время получилось дойти до **0.9333** — это +0.7 п.п. к baseline TextCNN из семинара (0.926) и около +1 п.п. к BiLSTM из семинара (0.92). До целевых 0.95 не дотянул.

Чтобы добраться до 0.95, надо одно из:
- GPU + 15–20 эпох (на CPU 10 эпох hybrid заняли ~27 мин, на GPU было бы 1–2 мин)
- Эмбеддинги побольше: GloVe-300 (370 МБ) или word2vec-google-news-300 (1.6 ГБ)
- EDA-аугментация (synonym replacement / random swap)
- Замена backbone на трансформер (BERT-base уверенно даёт 0.95+, но это уже не TextCNN/BiLSTM)

## Как запустить

```bash
pip install -r requirements.txt
jupyter notebook lesson2/homeworks/
```

Зависимости — в [`requirements.txt`](requirements.txt). При первом запуске `gensim.downloader` подкачает glove-100 (~128 МБ) в `~/gensim-data/` — это занимает несколько минут.
