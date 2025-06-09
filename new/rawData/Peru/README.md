# Notes on the Peru Data so far:


## Video Discovery and Assignment:
0 clips: 17 rows
1 clip: 100 rows
2 clips: 159 rows
3 clips: 69 rows
4 clips: 18 rows

246 rows out of 363 total rows (67.8%) have at least two video clips

we drop the 17 that have no clips

then we drop the 100 that have only one clip

we reconcile the 3 and 4 clip ones by finding the first and last clip and using those as the two clips

ones with exactly 2 clips did not need any processing and could be used as is

## Audio Conversion

- Some clips failed to convert:
- 5 rows were removed due to failure to convert to audio. 
- 6 rows removed because they were not in Spanish
- 4 rows were removed because the audio was too poor quality (per the evaluator comments)
- 1 row was removed because the audio was too short (4 minutes)
- 
## Transcription

- 4 rows were removed due to failure to process their transcriptions

## Cleaning

From the evaluatons:
- remove punctuation
- remove special characters
- remove extra whitespace
- only keep information that is L, M, H, N/A, or a single digit
- if data is missing or of another type, replace with N/A

Rows removed:
- rows where the audio clip is noted to be poor quality

Columns removed:
- removed extranous or unused columns


## Validation:

- Accuracy compared to the human evaluators is defined as $1-d$ where $d$ is the normalized distance between the human and AI scores.

## 5.5 Cleaning;

We justify some cleaining:

https://chatgpt.com/c/68471a79-6f30-8006-a2b9-aa2aec8a2c05


## 6 Performance: 

We need to measure:

---

Time on Learning: exact agreement on 2 of 3 snapshots.

Quality Elements: within 1 point of “master codes” on 8 of 9 high-inference elements per video.

Failing on the first set of three videos prompts feedback and a second attempt; failure on the second attempt means no certification. 

---

^^ If the AI is able to pass this exam, it should be considered certified and good enough to be used as a substitute for human evaluation. We should run this test for each of the models we are considering in the performance notebook. We should find a valid comparison between the human exam and the AI exam. Options:
- We could adminster a random selection of two sets of three observations from the test set to each model and see if it passes on the first set (first try) and if it needs to be re-administered on the second set (second try).
- We could take the average performance across all observations in the test set and see if it passes overall on the exam on average. 
