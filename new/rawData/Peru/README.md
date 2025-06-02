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



## Cleaning

From the evaluatons:
- remove punctuation
- remove special characters
- remove extra whitespace
- only keep information that is L, M, H, N/A, or a single digit
- if data is missing or of another type, replace with N/A

Rows to remove:
- non spanish rows and bilingual classes
- rows where the audio clip is noted to be poor quality

Columns to clean up:
- remove extranous or unused columns