# Course data

Obtain the instructor-assigned 50,000-review sample from the team/instructor. The full upstream dataset is Amazon Fine Food Reviews; the specific course sample has not been provided here.

Store the agreed CSV in this local directory. Raw data are ignored by Git. Record the shared filename, checksum, preprocessing, deduplication rule, split seed, and train/test row identifiers when the data owner implements the training pipeline. Do not commit raw review CSVs, user/profile identifiers, or local credentials.

The charter reports a random sample seed of 13 and three classes derived from `Score`: 1–2 negative, 3 neutral, 4–5 positive. These are team plans/reported statistics, not newly verified measurements. Use `Score` only to construct labels. Recalculate class distribution and majority baseline after deduplication/splitting.

The 18 invented demonstration reviews are embedded in `sentiment.py`; they are not a replacement for the assigned dataset.
