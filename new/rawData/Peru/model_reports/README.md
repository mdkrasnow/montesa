# Evaluation pipeline results and notes

- Using gemini 2.5 flash does not seem to yield any better results than using gemini 2.0 flash; we can continue to use gemini 2.0 flash for the intelligent model
- Marginally better performance on no-NA data than on full data
- Failing hard on the snapshots; might not be identifying properly; we should look into this with the following models:
  - Into the transcripts, bake the time stamp information into the transcript; this will allow the model to better understand the context and properly identify the snapshots
  - Multi-modal outputs
  - Using a reasoning model



# Models Descriptions:
- 1-BaseEvaluator: The base evaluator, using the Teach framework, the original Teach_1.json allowign for N/A values, and using gemini-2.0-flash for the LLM. The base model uses Chain-of-Thought reasoning to generate its responses before assigning a score and uses contrained generation to allow for consistent parsing of information.
- 2-BaseEvaluator-Validate: The base evaluator, using the Teach framework and the second version of the Teach_1.json. The changes made were increased rubric quality. We improved rubric quality by including the historical distribution of scores for each column to each rubric from the training data. Validation improvements were observed. The improvements we observe in validation might be greater than in the test set because the distribution of scores in the test set may technically be different from the training set.
- 3-BaseEvaluator-Validate-no-NA: The base evaluator, using the Teach framework and the newly improved Teach_1.json, using the original dataset. The changes made were increased rubric quality and removing N/A as an option; the justification for this being that N/As are rare in the data and the loss from a misclassified N/A is large, a well-trained model would learn to avoid N/As.
- 4-BaseEvaluator-Validate-no-NA-data: The base evaluator, using the Teach framework and the newly improved Teach_1.json, but only using the data that has dropped low-quality transcripts. I hypothesize that low quality data is a major source of error in the evaluateion data, and removing it will yield better results. This was anecdotally substantiated in the validation data, where the model performed marginally better on the no-NA data (drop the lowest ~14th percentile for transcript length)(FINAL_peru_cleaned_transcripts.csv) vs the original dataset (peru_cleaned_transcripts.csv).
- 5-BaseEvaluator-Validate-no-NA-data-g-25-flash: The base evaluator, using the Teach framework and the newly improved Teach_1.json, but only using the data that has dropped low-quality transcripts. Instead of using gemini-2.0-flash for the LLM, we use gemini-2.5-flash. We observe marginal improvements over the previous model, but it is not clear if this is due to the LLM or the validation data. 
