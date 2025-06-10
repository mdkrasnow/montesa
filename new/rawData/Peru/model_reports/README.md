# Evaluation pipeline results and notes

- Using gemini 2.5 flash does not seem to yield any better results than using gemini 2.0 flash; we can continue to use gemini 2.0 flash for the intelligent model
- Marginally better performance on no-NA data than on full data
- Failing hard on the snapshots; might not be identifying properly; we should look into this with the following models:
  - Into the transcripts, bake the time stamp information into the transcript; this will allow the model to better understand the context and properly identify the snapshots
  - Multi-modal outputs
  - Using a reasoning model