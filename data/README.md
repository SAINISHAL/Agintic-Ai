# Datasets (place files here)

Drop the official CSVs into these folders. Do not rename columns.

## Counselling

Put these files in `data/counselling/`:

- `my_dataset_train.csv`
- `my_dataset_val.csv`
- `my_dataset_test.csv`

Expected columns:

- `ID`
- `Speaker`
- `Utterance`
- `Code-Mix`
- `Emotion`
- `Dialogue_Act`
- `Sub topic`

## Bhagavad Gita Q&A

Put chapter files in `data/gita/`:

- `Chapter_1_QA.csv` … `Chapter_18_QA.csv`

Expected columns:

- `chapter`
- `verse_source`
- `question`
- `answer`

After copying files, run:

```bash
python scripts/prepare_emotion_data.py
python scripts/prepare_qwen_data.py
```
