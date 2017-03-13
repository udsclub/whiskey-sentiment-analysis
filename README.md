# Experiments with full rt + imdb datasets

### 1. Results

RANDOM_SEED = 42

| Approach| Train Accuracy| Validation Accuracy|Epochs|MAX WORDS|MAX SEQUENCE LENGTH|LSTM output|Droupout before LSTM cell|Droupout after LSTM|dropout_U|dropout_W|
|--------|:------:|:------:|:----:|:-----:|:---:|:---:|:----:|:---:|:----:|:----:|
| BLSTM   | 87.23% |84.60%| 21   |20000  | 70  | 256 |0 |0.2  | 0.2  |0.2 |
| BLSTM   | 86.41% |84.88%| 25   |20000  | 70  | 256 |0.2 |0.2  | 0.2  |0.2 |
| LSTM   | 86.07% |85.11%| 25   |20000  | 70  | 128 |0.2 |0.2  | 0.2  |0.2 |
| LSTM   | 85.46% |**85.39%**| 77   |20000  | 100  | 128 |0.2 |0.2  | 0.2  |0.2 |
| LSTMm   | 83.39% |84.40* | 58   |20000  | 100  | 128 |0.2 |0.2  | 0.2  |0.2 |
