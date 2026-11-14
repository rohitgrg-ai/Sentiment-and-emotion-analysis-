# Emotion, Sentiment and Sarcasm Detection for Social Media Posts

A natural language processing project that reads a short social media post and answers three questions:

1. **Which emotion does it express?** Anger, fear, joy, love, sadness, surprise or neutral.
2. **Is it positive, negative or neutral?**
3. **Could it be sarcastic?**

It comes with a Streamlit web app: paste a post, or pick one from the example wall, and see the reading along with how it was decided.

| Post | Emotion | Sentiment | Sarcasm |
|---|---|---|---|
| I just love it when my phone dies at 2% | Anger | Negative | 99% |
| I got the job!!! | Joy | Positive | 2% |
| my hands won't stop shaking before this interview | Fear | Negative | 14% |
| The meeting is at 3pm on Tuesday. | Neutral | Neutral | 1% |

The first post shows the core problem. It contains the word *love*, but the writer is annoyed. A model that only counts words reads it as joy.

## Contents

- [Results](#results)
- [How it works](#how-it-works)
- [What the data exploration found](#what-the-data-exploration-found)
- [Graphs](#graphs)
- [Getting started](#getting-started)
- [Rebuilding everything from scratch](#rebuilding-everything-from-scratch)
- [Datasets](#datasets)
- [Model and training details](#model-and-training-details)
- [Experiments that were rejected](#experiments-that-were-rejected)
- [Limitations](#limitations)
- [Project layout](#project-layout)
- [References](#references)

## Results

All scores come from test data that was never used for training or for any tuning decision.

| What was tested | Rows | Result |
|---|---|---|
| Emotion in Reddit comments (GoEmotions) | 3,799 | Accuracy 0.728, macro-F1 0.688 |
| Emotion in tweets (SemEval-2018 Affect in Tweets) | 1,606 | Accuracy 0.757, macro-F1 0.531 |
| Sarcasm (iSarcasmEval) | 1,394 | Precision 0.705, recall 0.335, F1 0.454 |
| Irony (SemEval-2018) | 784 | Precision 0.783, recall 0.511, F1 0.619 |
| Sentiment (TweetEval) | 12,280 | Accuracy 0.543, macro-F1 0.543 |

**Macro-F1** averages the F1 score of every class equally, so a rare emotion counts as much as a common one. That is why tweet macro-F1 (0.531) sits well below tweet accuracy (0.757): love and surprise are rare in tweets and hard to detect.

![F1 score for each emotion](report/figures/results_per_emotion_f1.png)

### RoBERTa against the TF-IDF baseline

The project started with a TF-IDF + Linear SVM model. On its own test split it looked excellent, but it did not hold up on real posts.

| Test | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| Baseline's own test split, emotion | TF-IDF + SVM | 0.963 | 0.963 |
| Baseline's own test split, sentiment | TF-IDF + SVM | 0.946 | 0.946 |
| Same 2,195 GoEmotions comments, six emotions | TF-IDF + SVM | 0.305 | 0.286 |
| Same 2,195 GoEmotions comments, six emotions | RoBERTa | 0.761 | 0.771 |

Both models are scored on the same comments, leaving out neutral because the baseline has no neutral class.

![Both models on the same comments](report/figures/results_baseline_vs_roberta.png)

The baseline stays in the project for comparison. The app's "How this was decided" panel shows its answer, but RoBERTa makes every final decision.

## How it works

![How a post becomes a final answer](report/figures/method_pipeline.png)

1. **Clean the post.** Usernames become `@user` and links become `http`, matching how the base model was pretrained.
2. **Read it with RoBERTa.** One shared encoder feeds two outputs: scores for seven emotions, and the probability that the post is sarcastic.
3. **Add the emoji vote.** Known emojis vote for an emotion, with a weight of 0.25. A post with only emojis uses the vote alone.
4. **Apply the sarcasm rule.** If sarcasm is 90% or more and the top emotion is joy or love, the system uses the next most likely emotion, leaving out neutral.
5. **Work out sentiment from the emotion.** Joy and love are positive. Anger, fear and sadness are negative. Surprise and neutral are neutral.

The 0.90 threshold is the lowest value that costs no more than 0.005 emotion macro-F1 on validation data. Lower thresholds catch more sarcasm, but they also overturn more genuine joy and love.

![Choosing the sarcasm threshold](report/figures/results_threshold_choice.png)

## What the data exploration found

Exploring the data before training explained why the first model failed and shaped the rest of the project.

- **The baseline data does not look like social media.** 96.8% of its rows use "feel", "feels" or "feeling", as in "i feel really helpless". In real comments and tweets the figure is about 2%. A model trained on it learns to spot emotion adjectives, not to read posts.
- **The baseline data had label leakage.** 44,650 rows carried emojis that encoded the answer. Removing them left texts with conflicting labels, which were dropped.
- **Its sentiment labels came from a word list and often contradict the emotion.** 6,704 sadness rows are labelled positive, and 5,020 joy rows are labelled negative. This is why the app works out sentiment from the emotion instead.
- **Real data is unbalanced.** Neutral is 42.6% of GoEmotions, and love is only 0.6% of the tweets.
- **Emojis matter in tweets.** About 15% of tweets contain an emoji, compared with 1.5% of Reddit comments and none of the cleaned baseline data.
- **Sarcasm data can leak too.** In SemEval irony, hashtags such as `#not` gave away the label, so they were removed.

![Writing style differences](report/figures/eda_writing_style.png)

## Graphs

Every graph here was drawn with matplotlib. `report/make_figures.py` redraws the 14 report graphs from the data and the trained model, and the full report is in [`report/`](report). Four of them also appear in the sections above.

### Data exploration

**How many rows each dataset has.** The raw baseline data is by far the largest, but most of it was removed in cleaning and balancing.

![Rows in each dataset](report/figures/eda_dataset_sizes.png)

**Cleaning the baseline data.** Most rows survive cleaning, and the big drop comes from balancing the classes.

![Rows left after each cleaning step](report/figures/eda_cleaning_steps.png)

**How the emotions are spread.** Balancing gave the baseline data 16.7% per emotion. The real datasets are very uneven.

![Share of each emotion in each dataset](report/figures/eda_emotion_balance.png)

**How long the posts are.** The median post has 12 to 19 words, so a 64-token limit covers almost every post.

![Post length in words](report/figures/eda_text_length.png)

**Sarcastic and plain rows.** iSarcasm has far more plain tweets than sarcastic ones, while SemEval irony is close to even.

![Sarcastic and non-sarcastic rows by source](report/figures/eda_sarcasm_balance.png)

**Emotion against word-list sentiment.** The baseline's sentiment labels often contradict the emotion, such as sadness labelled positive.

![Emotion against word-list sentiment](report/figures/eda_emotion_vs_sentiment.png)

### Results

**Final test scores.** Longer bars are better, and 1.0 is perfect.

![Final test scores](report/figures/results_overview.png)

**Where the answers go.** Each row shows what the model said for posts with that true emotion, as a percentage. Dark squares on the diagonal are correct.

![Confusion matrices for comments and tweets](report/figures/results_confusion_matrices.png)

**What the TF-IDF baseline learned.** Its strongest words are adjectives that finish "i feel ...", which real posts rarely use.

![Top words for each emotion in the baseline](report/figures/results_baseline_keywords.png)

<details>
<summary><strong>Earlier experiments: TF-IDF baseline and emoji handling (16 graphs)</strong></summary>

These graphs come from the project's first stage, before the RoBERTa model. They compare six ways of handling emojis with the TF-IDF model:

| System | What it does |
|---|---|
| A | Words only |
| B | Emojis turned into their names |
| B' | B, also trained on extra emoji examples |
| C | Separate emoji features added |
| C' | C, also trained on extra emoji examples |
| D | A, plus the emoji vote |

Except for the first two, these graphs use a hand-built set of 500 ambiguous posts. Its labels were suggested by AI and have not been checked by a person, so treat these results as provisional.

**Baseline on its own test split.** Almost every answer is correct here, which is the score that did not transfer to real posts.

![Baseline emotion confusion matrix](report/figures/emotion_confusion_matrix.png)
![Baseline sentiment confusion matrix](report/figures/sentiment_confusion_matrix.png)

**Accuracy by system on the ambiguous set.**

![Accuracy by subset](report/figures/accuracy_by_subset.png)

**Sensitivity against stability.** Does a system change its answer when the emoji should change it, and keep it when it should not?

![Sensitivity against stability](report/figures/paired_sensitivity_vs_stability.png)

**Emotion confusion matrices on the ambiguous set.**

![Emotion, system A](report/figures/confusion_emotion_A.png)
![Emotion, system B](report/figures/confusion_emotion_B_orig.png)
![Emotion, system B'](report/figures/confusion_emotion_B_aug.png)
![Emotion, system C](report/figures/confusion_emotion_C_orig.png)
![Emotion, system C'](report/figures/confusion_emotion_C_aug.png)
![Emotion, system D](report/figures/confusion_emotion_D.png)

**Sentiment confusion matrices on the ambiguous set.**

![Sentiment, system A](report/figures/confusion_sentiment_A.png)
![Sentiment, system B](report/figures/confusion_sentiment_B_orig.png)
![Sentiment, system B'](report/figures/confusion_sentiment_B_aug.png)
![Sentiment, system C](report/figures/confusion_sentiment_C_orig.png)
![Sentiment, system C'](report/figures/confusion_sentiment_C_aug.png)
![Sentiment, system D](report/figures/confusion_sentiment_D.png)

</details>

## Getting started

### Requirements

- Python 3.9 or later
- About 500 MB of disk space for the trained model
- A GPU is optional for running the app, but strongly recommended for training

### Install

```bash
git clone https://github.com/rohitgrg-ai/Sentiment-and-emotion-analysis-.git
cd Sentiment-and-emotion-analysis-
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Add the trained model

The trained model is too large for GitHub, so it is not in this repository. Put its files in `models/emotion_sarcasm/`:

```
models/emotion_sarcasm/
    config.json
    model.safetensors
    merges.txt
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
    vocab.json
```

To show the baseline comparison in the app, also put `tfidf_vectorizer.joblib`, `emotion_linear_svm.joblib` and `sentiment_linear_svm.joblib` in `models/`. Without them the app still works and hides that line.

### Run the app

Run it from the project folder:

```bash
python -m streamlit run src/app.py
```

Then open the address Streamlit prints, usually http://localhost:8501.

## Rebuilding everything from scratch

### 1. Download the raw data

`src/build_data.py` expects a folder containing:

| Path | Source |
|---|---|
| `irony/{train,validation,test}.parquet` | SemEval-2018 Task 3, from `cardiffnlp/tweet_eval` (config `irony`) |
| `isarcasm/train.En.csv`, `isarcasm/task_A_En_test.csv` | [iSarcasmEval](https://github.com/iabufarha/iSarcasmEval) |
| `ait2018/all-data.zip` | SemEval-2018 Task 1 (Affect in Tweets) |

GoEmotions downloads automatically through the `datasets` package.

### 2. Build the datasets

```bash
python src/build_data.py --raw path/to/raw
```

This writes the train, validation and test files for comment emotion, tweet emotion and sarcasm into `data/`. It removes any text that appears in more than one split.

### 3. Train the model in two passes

```bash
python src/train.py --base cardiffnlp/twitter-roberta-base-2022-154m
python src/label_sarcasm_emotions.py --raw path/to/raw/isarcasm/train.En.csv
python src/train.py --base cardiffnlp/twitter-roberta-base-2022-154m --extra-emotion data/sarcasm_emotion_pseudo.csv
```

iSarcasmEval authors also wrote what they really meant. The first pass trains a model, which then reads each author's plain rephrase and gives the sarcastic tweet that emotion. The second pass trains the final model with those extra labels. Each pass saves the model to `models/emotion_sarcasm/` and its test scores to `results/metrics.json`.

To score an existing model without training, add `--eval-only`. For a quick check on a few rows, add `--limit 200`.

### 4. Train the baseline

Put the raw baseline dataset at `data/text2.csv`, then run:

```bash
python src/balance_dataset.py
python src/baseline.py
```

The first command removes leakage and duplicates, then balances emotion and sentiment at the same time. It writes `data/final.csv` and `data/balancing_summary.json`. The second trains the TF-IDF + Linear SVM models and writes `results/baseline_metrics.json`.

### 5. Redraw the report graphs

```bash
python report/make_figures.py --data data
```

This needs the trained model, the baseline, the files from steps 2 and 4, and the TweetEval sentiment test set saved as `data/natural_sentiment_test.csv` with `text` and `sentiment` columns. It saves the graphs to `report/figures/` and every number they show to `results/report_numbers.json`. Add `--cache some/folder` to save the model's outputs, so later runs are faster.

## Datasets

| Dataset | Rows | Used for | Labelled by |
|---|---|---|---|
| Baseline data (`text2.csv`, `final.csv`) | 416,809 raw, 58,980 balanced | Training the TF-IDF baseline | Emotion came with the data. Sentiment came from a word list. |
| [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (Reddit comments) | 37,465 | Training and testing emotion | People |
| SemEval-2018 Affect in Tweets | 5,521 | Training and testing emotion | People |
| iSarcasmEval and SemEval-2018 Irony | 10,214 | Training and testing sarcasm | Tweet authors (iSarcasm) and people (irony) |
| TweetEval sentiment | 12,280 | Testing sentiment only | People |

GoEmotions has 27 emotions. They were grouped into the project's seven, for example admiration and gratitude into love, and annoyance and disgust into anger. The full mapping is `GO_EMOTIONS` in `src/build_data.py`.

The datasets are not stored in this repository. Check each dataset's terms before sharing it.

## Model and training details

| Setting | Value |
|---|---|
| Base model | `cardiffnlp/twitter-roberta-base-2022-154m`, a RoBERTa model pretrained on tweets |
| Outputs | Emotion (7 classes) and sarcasm (2 classes) from one shared encoder |
| Training rows | 29,920 comments, 3,485 emotion tweets, 6,736 sarcasm rows, and 462 sarcastic tweets with author-based emotion labels counted 3 times |
| Learning rate | 0.00002, with linear warm-up over the first 6% of steps |
| Epochs, batch size | 3 epochs, 32 posts |
| Maximum post length | 64 tokens |
| Class weights | Square root of inverse class frequency, so rare emotions count more |
| Model selection | The epoch with the best validation score, the mean of emotion macro-F1 and sarcasm F1 |
| Random seed | 42 |

Each training row only updates the output it has a label for. GoEmotions comments also count as not sarcastic, which stops the model from flagging ordinary text.

## Experiments that were rejected

Each experiment had pass rules written down before it ran. A change had to improve validation results without lowering either emotion test score.

| Experiment | Result | Decision |
|---|---|---|
| Adjust each emotion's score separately | GoEmotions rose from 0.688 to 0.708, but tweets fell from 0.531 to 0.529 | Rejected |
| Retrain with 9,475 extra rows carrying mixed emotion labels | Validation, GoEmotions and tweet scores all fell. Only iSarcasm F1 improved, to 0.485. | Rejected |

## Limitations

- The model misses about two-thirds of sarcasm on iSarcasmEval, and about 3 in 10 of its sarcasm flags there are wrong.
- Love and surprise in tweets score 0.13 and 0.10, and those scores come from only 10 and 17 test rows.
- Sentiment is worked out from the emotion, so it is only as good as the emotion prediction. It is not trained on sentiment labels directly.
- The emoji vote uses a fixed table, so an emoji counts the same whatever the post says.
- Some sarcastic training labels come from the first model, so some are wrong. Irony labels stand in for sarcasm in the SemEval rows.
- English only.

### Possible next steps

- Collect more tweets showing love and surprise.
- Train a sentiment output directly on labelled sentiment data.
- Let the emoji vote depend on the words around each emoji.

## Project layout

| Path | Purpose |
|---|---|
| `src/app.py`, `src/app_style.css` | Streamlit app and its styling |
| `src/model.py` | The model, emoji vote, sarcasm rule and prediction logic |
| `src/train.py` | Training and test scoring |
| `src/build_data.py` | Builds the comment, tweet and sarcasm datasets |
| `src/label_sarcasm_emotions.py` | Emotion labels for sarcastic tweets, from their authors' rephrases |
| `src/balance_dataset.py` | Cleans and balances the baseline data |
| `src/baseline.py` | TF-IDF + Linear SVM baseline |
| `src/*.ipynb` | Early data exploration notebooks |
| `report/` | Project report in Word and PDF, its graphs, and `make_figures.py` |
| `results/` | Test scores in JSON |
| `data/balancing_summary.json` | Counts from each step of cleaning the baseline data |
| `.streamlit/config.toml` | App theme |

## References

- Abu Farha, I., Oprea, S., Wilson, S., and Magdy, W. (2022). SemEval-2022 Task 6: iSarcasmEval, Intended Sarcasm Detection in English and Arabic. *Proceedings of SemEval-2022.*
- Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., and Neves, L. (2020). TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification. *Findings of EMNLP 2020.*
- Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., and Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. *Proceedings of ACL 2020.*
- Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv:1907.11692.*
- Loureiro, D., Barbieri, F., Neves, L., Espinosa Anke, L., and Camacho-Collados, J. (2022). TimeLMs: Diachronic Language Models from Twitter. *Proceedings of ACL 2022, System Demonstrations.*
- Mohammad, S., Bravo-Marquez, F., Salameh, M., and Kiritchenko, S. (2018). SemEval-2018 Task 1: Affect in Tweets. *Proceedings of SemEval-2018.*
- Van Hee, C., Lefever, E., and Hoste, V. (2018). SemEval-2018 Task 3: Irony Detection in English Tweets. *Proceedings of SemEval-2018.*
