# Pipeline specifications


## Base Gemini Model: 1-BaseEvaluator_evaluations

Dataset: **peru_cleaned_transcripts.csv**

Teach Framework Rubric: **low_Teach_1.json**

Model: gemini-2.0-flash


## Improved Gemini Model: 2-BaseEvaluator-Validate

Dataset: **peru_cleaned_transcripts.csv**

Teach Framework Rubric: **med_Teach_1.json**

Model: gemini-2.0-flash


## Improved Gemini Model improved JSON: 3-BaseEvaluator-Validate

Dataset: **peru_cleaned_transcripts.csv**

Teach Framework Rubric: **high_Teach_1.json**

Model: gemini-2.0-flash


## Improved Gemini Model with improved JSON without NAs: 4-BaseEvaluator-Validate-no-NA-data

Dataset: **RMNA_cleaned_transcripts.csv**

Teach Framework Rubric: **high_Teach_1.json**

Model: gemini-2.0-flash


## Improved Gemini Model with improved JSON without NAs using Gemini 2.5 flash: 5-BaseEvaluator-Validate-no-NA-data-Gemini-2.5-flash

Dataset: **RMNA_cleaned_transcripts.csv**

Teach Framework Rubric: **high_Teach_1.json**

Model: gemini-2.5-flash


## Improved Gemini Model with improved JSON using Gemini 2.0 flash with formatted transcripts: 6-BaseEvaluator-Validate-timestamped-data-g-20-flash

Dataset: **TIMESTAMPED_cleaned_transcripts.csv**

Teach Framework Rubric: **high_Teach_1.json**

Model: gemini-2.0-flash

