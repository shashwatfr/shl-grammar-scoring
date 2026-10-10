# Grammar scoring from spoken audio

My entry for the SHL hiring assessment 2026 on Kaggle (`shl-hiring-assessment-2026`, Kaggle username `shashwatfr`). The task is to predict a grammar score from 0 to 5 for a short spoken clip. The metric is RMSE.

Everything is in `grammar_scoring.ipynb`. It runs top to bottom in Colab on a T4 GPU. The report (approach, preprocessing, pipeline, evaluation) is in the notebook's markdown cells, and the outputs are saved in the notebook.

## Approach in short

- Pretrained Whisper encoders (small and medium) are used frozen. For each hidden layer I take the mean and std of the frame vectors over time, which gives one fixed-size vector per clip.
- Whisper small also transcribes each clip. From the transcript I compute a few fluency statistics plus GPT-2 token negative log-likelihood.
- One ridge regression per feature group (standardised inputs, alpha picked by CV), blended with fixed weights 5 : 2 : 1 (Whisper medium, Whisper small, text) and clipped to [0, 5].

Only 769 labelled clips are available, so I kept the learned part linear and small.

## Results

| | RMSE |
|---|---|
| constant prediction (mean label) | 1.014 |
| training-data RMSE (fit and scored on the non-zero training rows, optimistic) | 0.349 |
| 5-fold CV, non-zero rows | 0.521 |
| 5-fold CV, ids 0-215 (closest to the test set) | 0.545 |
| public leaderboard | 0.4018 |

The 37 zero-label clips come from a different recording batch, so I exclude them from training and from the headline metrics. Scoring them as well gives a training-data RMSE of 0.652, because the model never saw them.

The CV numbers are the honest estimate. The hyperparameters were chosen on the same folds, so they are slightly optimistic, and the in-sample number shouldn't be read as quality. Whisper medium does most of the work; dropping it costs about 0.036 CV RMSE, while dropping Whisper small or the text group costs about 0.002 (within noise).

The saved outputs were produced from cached features. Rebuilding every feature from scratch with the notebook's own extraction code gave CV RMSE within about 0.002 of the numbers above.

## Running it

1. Open the notebook in Colab with a GPU runtime.
2. Add a Colab secret named `KAGGLE_KEY` holding a Kaggle API token (the new `KGAT_` kind) and accept the competition rules on Kaggle. Outside Colab, set `KAGGLE_API_TOKEN`, or point `SHL_ROOT` at an existing copy of the data.
3. Run all cells. Extraction takes a while the first time; features are cached in `cache/` and reused after that.
4. `submission.csv` is written at the end, keyed by `test.csv` (216 rows). `sample_submission.csv` in the competition data has a different row count and shouldn't be used.

## Models and licences

| model | licence (from its model card) |
|---|---|
| `openai/whisper-small`, `openai/whisper-medium` | Apache-2.0 |
| `gpt2` | MIT |

I also tried `microsoft/wavlm-base-plus` while developing. It improved CV RMSE slightly (0.521 to 0.513) and public score slightly (0.3956 vs 0.4018), but its model card points to a CC BY-SA licence, so I left it out of the final pipeline.

## Not included

Competition audio and csvs, and the cached features, are not in the repo (competition data can't be redistributed). The `.gitignore` covers them.
