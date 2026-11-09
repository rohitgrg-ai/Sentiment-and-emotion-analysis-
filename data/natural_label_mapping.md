# Label mapping (fixed before any model was trained on this data)

GoEmotions label -> project emotion. Only comments with exactly one original label are kept.

| project emotion | GoEmotions labels |
|---|---|
| joy | joy, amusement, excitement, relief, pride |
| love | love, caring, gratitude, admiration |
| anger | anger, annoyance, disgust, disapproval |
| fear | fear, nervousness |
| sadness | sadness, grief, disappointment, remorse |
| surprise | surprise |
| neutral | neutral |

Dropped (no clean match, never forced into an emotion): approval, confusion, curiosity, desire, embarrassment, optimism, realization.

Sentiment: TweetEval labels used as given (0 negative, 1 neutral, 2 positive).

'neutral' is an extra emotion class here; the project's earlier data had only six emotions. Decide and document how it is evaluated against the ambiguous set (whose emotion labels have no neutral).