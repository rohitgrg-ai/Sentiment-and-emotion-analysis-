# Natural dataset build report

Sources: GoEmotions (emotion), TweetEval sentiment. Human-labelled; check licences before redistributing. final.csv untouched.

## Emotion rows kept per split

| split | rows | anger | fear | joy | love | neutral | sadness | surprise |
|---|---|---|---|---|---|---|---|---|
| train | 29920 | 4362 | 512 | 3127 | 6517 | 12774 | 1911 | 717 |
| val | 3746 | 546 | 66 | 382 | 845 | 1591 | 221 | 95 |
| test | 3799 | 596 | 77 | 348 | 851 | 1604 | 236 | 87 |

## Emotion filtering

{'train': {'raw_rows': 43410, 'single_label_mapped': 30149, 'multi_label_dropped': 7102, 'unmapped_label_dropped': 6159}, 'val': {'raw_rows': 5426, 'single_label_mapped': 3759, 'multi_label_dropped': 878, 'unmapped_label_dropped': 789}, 'test': {'raw_rows': 5427, 'single_label_mapped': 3806, 'multi_label_dropped': 837, 'unmapped_label_dropped': 784}}

## Emotion overlap / duplicate removal

{'train': {'rows_before': 30149, 'removed_ambiguous_overlap': 2, 'removed_duplicates': 171, 'removed_in_later_split': 56, 'rows_after': 29920}, 'val': {'rows_before': 3759, 'removed_ambiguous_overlap': 1, 'removed_duplicates': 4, 'removed_in_later_split': 8, 'rows_after': 3746}, 'test': {'rows_before': 3806, 'removed_ambiguous_overlap': 1, 'removed_duplicates': 6, 'removed_in_later_split': 0, 'rows_after': 3799}}

## Sentiment rows kept per split

| split | rows | negative | neutral | positive |
|---|---|---|---|---|
| train | 45585 | 7092 | 20654 | 17839 |
| val | 2000 | 312 | 869 | 819 |
| test | 12280 | 3970 | 5935 | 2375 |

## Sentiment overlap / duplicate removal

{'train': {'rows_before': 45615, 'removed_ambiguous_overlap': 0, 'removed_duplicates': 30, 'removed_in_later_split': 0, 'rows_after': 45585}, 'val': {'rows_before': 2000, 'removed_ambiguous_overlap': 0, 'removed_duplicates': 0, 'removed_in_later_split': 0, 'rows_after': 2000}, 'test': {'rows_before': 12284, 'removed_ambiguous_overlap': 0, 'removed_duplicates': 4, 'removed_in_later_split': 0, 'rows_after': 12280}}

Notes: emojis and slang are kept in the text. 'removed_ambiguous_overlap' counts rows matching the 500-row ambiguous test set; it must never be trained on.