# Emotion and Sarcasm Detection for Social Media Posts

Paste a short post and this Streamlit app shows its emotion, its sentiment, and whether it looks sarcastic.

## How it works

One Twitter-based RoBERTa model (`cardiffnlp/twitter-roberta-base-2022-154m`) has two outputs:

- **Emotion:** anger, fear, joy, love, sadness, surprise, or neutral.
- **Sarcasm:** the probability that a post is sarcastic.

Known emojis add a vote to the emotion score, with a weight of 0.25. If a post is very likely sarcastic (90% or higher) and its top emotion is joy or love, the app uses the next most likely emotion instead. Sentiment then follows from the emotion: joy and love are positive, anger, fear and sadness are negative, and surprise and neutral are neutral.

## Run the app

Python 3.9 or later. The trained model must be in `models/emotion_sarcasm/`; see below to train it.

```bash
pip install -r requirements.txt
python -m streamlit run src/app.py
```

## Results

| Test set | Rows | Result |
|---|---|---|
| Emotion, GoEmotions | 3,799 | Accuracy 0.728, macro-F1 0.688 |
| Emotion, SemEval-2018 Affect in Tweets | 1,606 | Accuracy 0.757, macro-F1 0.531 |
| Sarcasm, iSarcasmEval | 1,394 | Precision 0.705, recall 0.335, F1 0.454 |
| Irony, SemEval-2018 | 784 | Precision 0.783, recall 0.511, F1 0.619 |
| Sentiment, TweetEval | 12,280 | Accuracy 0.543, macro-F1 0.543 |

Full emotion and sarcasm scores are in `results/metrics.json`. The numbers behind the report's graphs are in `results/report_numbers.json`.

## Baseline model

Before the transformer, the project trained a TF-IDF + Linear SVM on a cleaned, balanced dataset. It stays in the project as a baseline, and the app's "How this was decided" panel shows its answer for comparison. The transformer makes every final decision.

| Baseline test | Accuracy | Macro-F1 |
|---|---|---|
| Emotion, its own test split | 0.9627 | 0.9625 |
| Sentiment, its own test split | 0.9456 | 0.9455 |
| Emotion, human-labelled GoEmotions text (six emotions) | 0.3048 | 0.2863 |

The high scores did not transfer: most training sentences follow an "i feel X" pattern, so the SVM mainly learned emotion keywords. That result is why the project moved to a transformer trained on natural comments and tweets.

## Limitations

- The model misses about two-thirds of sarcasm on iSarcasmEval, and about 3 in 10 of its sarcasm flags there are wrong.
- The emoji vote uses a fixed table, so an emoji counts the same whatever the post says.
- Sentiment is a mapping from emotion, so it is only as good as the emotion prediction.
- Some sarcastic training labels are machine-generated, and irony labels stand in for sarcasm in SemEval rows.
- English only.

## Train the model

The model and tweet datasets are not stored in this repository. Download the raw files listed at the top of `src/build_data.py`, then run:

```bash
python src/build_data.py --raw path/to/raw
python src/train.py --base cardiffnlp/twitter-roberta-base-2022-154m
python src/label_sarcasm_emotions.py --raw path/to/raw/isarcasm/train.En.csv
python src/train.py --base cardiffnlp/twitter-roberta-base-2022-154m --extra-emotion data/sarcasm_emotion_pseudo.csv
```

The first training pass labels sarcastic tweets using their authors' plain rephrases. The second pass trains the final model on those labels too. A GPU is strongly recommended.

To train the baseline, put the raw dataset at `data/text2.csv`, then run:

```bash
python src/balance_dataset.py
python src/baseline.py
```

## Project layout

| Path | Purpose |
|---|---|
| `src/app.py`, `src/app_style.css` | Streamlit app and styling |
| `src/model.py` | Model and prediction logic |
| `src/train.py` | Training and evaluation |
| `src/build_data.py` | Dataset preparation |
| `src/label_sarcasm_emotions.py` | Emotion labels for sarcastic tweets |
| `src/balance_dataset.py`, `src/baseline.py` | Cleaned dataset and TF-IDF + SVM baseline |
| `src/*.ipynb` | Early data exploration notebooks |
| `report/` | Project report (Word and PDF), figures, and `make_figures.py`, which redraws the graphs |

## Data sources

GoEmotions, TweetEval, SemEval-2018 Tasks 1 and 3, and iSarcasmEval (SemEval-2022 Task 6). Check each dataset's terms before sharing the data.
