# Experiments with full rt + imdb datasets

### 1. Results

RANDOM_SEED = 42

| Approach| Train Accuracy| Validation Accuracy|Epochs|MAX WORDS|MAX SEQUENCE LENGTH|LSTM output|Droupout before LSTM cell|Droupout after LSTM|dropout_U|dropout_W|
|--------|:------:|:------:|:----:|:-----:|:---:|:---:|:----:|:---:|:----:|:----:|
| BLSTM   | 87.23% |84.60%| 21   |20000  | 70  | 256 |0 |0.2  | 0.2  |0.2 |
| BLSTM   | 86.41% |84.88%| 25   |20000  | 70  | 256 |0.2 |0.2  | 0.2  |0.2 |

