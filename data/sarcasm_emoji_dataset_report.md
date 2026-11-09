# Sarcasm and tweet-emotion dataset report

Labels: iSarcasm by the tweets' authors; SemEval-2018 irony and Affect in Tweets by annotators.
`sarcastic` uses irony as a stand-in for SemEval rows. Use `source` to train or score them separately.

## Sarcasm rows by label
```
           train  val  test
sarcastic                  
0           4517  757  1667
1           2219  543   511
```
## Sarcasm rows and sarcastic rate by source
```
train: {'isarcasm': {'size': 3099, 'mean': 0.25}, 'isarcasm_rephrase': {'size': 776, 'mean': 0.0}, 'semeval2018_irony': {'size': 2861, 'mean': 0.505}}
val: {'isarcasm': {'size': 346, 'mean': 0.251}, 'semeval2018_irony': {'size': 954, 'mean': 0.478}}
test: {'isarcasm': {'size': 1394, 'mean': 0.143}, 'semeval2018_irony': {'size': 784, 'mean': 0.397}}
```
Rows containing a label hashtag before removal: {'train': 1, 'val': 0, 'test': 431}

Rows with an emoji (%): {'train': 13.6, 'val': 12.4, 'test': 17.7}

## Sarcasm overlap removal
```
       rows_before  empty  ambiguous_overlap  conflicting_labels  duplicates  in_later_split  rows_after
train         6762      1                  0                  12          18               3        6736
val           1302      1                  0                   0           1               0        1300
test          2184      0                  0                   0           6               0        2178
```
## Tweet emotion rows
```
          train  val  test
emotion                   
anger      1284  144   525
fear        328   38   132
joy        1156  175   629
love         21    4    10
neutral     204   14    75
sadness     442   50   218
surprise     50    5    17
```
Rows with an emoji (%): {'train': 9.8, 'val': 23.3, 'test': 23.1}

## Tweet emotion filtering
```
       raw_rows  kept  dropped_mixed_or_no_home
train      6838  3517                      3321
val         886   430                       456
test       3259  1606                      1653
```
## Tweet emotion overlap removal
```
       rows_before  empty  ambiguous_overlap  conflicting_labels  duplicates  in_later_split  rows_after
train         3517      0                  0                   0          32               0        3485
val            430      0                  0                   0           0               0         430
test          1606      0                  0                   0           0               0        1606
```
Mapping: anger and disgust become anger; fear, joy, love, sadness and surprise keep their names.
anticipation, optimism, pessimism, trust are ignored beside one mapped emotion. Rows with only those labels are dropped.
Rows with two or more project emotions are dropped. Rows with no labels become neutral.