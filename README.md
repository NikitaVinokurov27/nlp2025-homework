# nlp2025-homework

Домашка по курсу [tam2511/nlp2025](https://github.com/tam2511/nlp2025).

## lesson2 — Классификация AG News

Цель — Macro F1 ≥ 0.95 на тесте.

Сделал два ноутбука, чтобы было видно прогрессию:

| Ноутбук | Модель | Test Macro F1 | Test Accuracy |
|---|---|---|---|
| [`agnews_textcnn.ipynb`](lesson2/homeworks/agnews_textcnn.ipynb) | TextCNN (kernels=2,3,4,5) + GloVe-100 | **0.9262** | 0.9263 |
| [`agnews_hybrid.ipynb`](lesson2/homeworks/agnews_hybrid.ipynb) | TextCNN + BiLSTM(hidden=64) + GloVe-100 | **0.9312** | 0.9313 |

Оба прогонялись локально на CPU (без GPU). В ноутбуках уже встроены выводы — графики обучения, classification report, confusion matrix и разбор ошибок.

**Что использовал из семинара**:
- структура TextCNN (kernels 3/4/5) — из `textcnn_agnews.ipynb`
- masked max-pool для BiLSTM — из `bilstm_agnews.ipynb`
- загрузка предобученных эмбеддингов через `gensim.downloader` — из `bilstm_word2vec.ipynb`

**Что добавил поверх семинара**:
- kernel=2 в CNN (помог немного)
- BiLSTM поверх тех же эмбеддингов, склейка с CNN-фичами
- AdamW + CosineAnnealingLR
- label smoothing 0.1
- BatchNorm перед классификатором
- gradient clipping 1.0
- токен `<num>` для чисел (улучшил покрытие словаря)

**Честно о результате**: на CPU за разумное время получилось дойти до 0.93 — это +1.5 п.п. к baseline TextCNN из семинара. Чтобы добраться до 0.95, по идее нужно одно из:
- GPU и 10–12 эпох (вместо 6)
- эмбеддинги побольше (GloVe-300 или word2vec-google-news-300 — 1.6 ГБ)
- EDA-аугментация (synonym replacement / random swap)
- замена back-end на transformer (BERT-base уверенно даёт 0.95+)

## Как запустить

```bash
pip install torch pytorch-lightning torchmetrics datasets gensim scikit-learn matplotlib seaborn jupyter

# или одной командой:
pip install -r requirements.txt

jupyter notebook lesson2/homeworks/
```

Зависимости перечислены в [requirements.txt](requirements.txt).
