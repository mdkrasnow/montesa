# Model Decisions

## What models will we use?
### To be used in both inter-rater reliability measurement and LLM-as-a-Judge evaluation, we will use the following models:
1. Eval by SwiftScore model with single-clip knowledge
2. Eval by SwiftScore model with partial two-clip knowledge using conversational style evaluation (human-style)
4. QLoRA model with single-clip knowledge
5. Agent group-chat model with single-clip knowledge
  
### To be used only in LLM-as-a-Judge evaluation:
5. Base Eval by SwiftScore model with full two-clip knowledge

## Eval by SwiftScore models
### These are different variations of the Eval by SwiftScore model, each with different parameters and configurations

We will first use DSPy to find optimal prompt (?)
Prompt engineering techniques:
- CoT
  - Specific ways that principals think about each one and analyzing in that way 
- Prompt chaining
  - Gathering context about the users

1. Reasoners:
    - o4-mini-high
    - gemini-2.5-flash-thinking
    - DeepSeek-R1
2. Non-reasoners:
    - GPT-4.1 / 4o
    - gemini-2.5-flash / gemini-2.0-flash
    - DeepSeek-V2

### We will test the effectiveness of providing the audio recording as well as the audio transcription to the models
1. We will do this with Gemini 2.5 Flash and Flash-thinking


---

# Model categories:
1. Base (Control)
   1. Experiment with model variants
2. Prompt Chaining
   1. Base variant, utilizing prompt chaining before scoring
3. Chain of Thought
   1. Base variant, utilizing chain of thought before scoring
4. Reasoner
   1. They will be prompt-optimized
   2. Experiment with model variants
6. Multimodal
   1. They will be prompt-optimized
   2. Experiment with model variants
7. QLoRA
   1. Trained on training data and hyperparameters trained via cross validation
8. Agents
   1. Experiment with different agent structures

---


### ! After intiial validation results, we will decide if we need more models / want to experiment with synthetic data






# Justification for Two-Clip, Two-Evaluation Approach

## 1. Overview
We have chosen to maintain a **one evaluation per clip** setup (i.e., two clips → two AI outputs) rather than collapsing both clips into a single evaluation. This decision is guided by principles of fairness, granularity, and direct comparability to human labels. Below, we document the rationale and planned comparison so that we can reference it in our notes and later justify it in our paper.

---

## 2. Rationale

### 2.1. Preserve Evaluation Granularity
- **Human Label Structure**  
  Human evaluators scored Clip 1 and Clip 2 separately, even though they were aware of both clips when annotating. Each clip received its own set of Teach‐dimension scores (e.g., clarity, engagement, pacing, etc.).  
- **AI Output Requirements**  
  To compute inter‐rater reliability (e.g., Cohen’s κ, MAE) between AI and human scores, the **unit of evaluation must match exactly**. Because human labels exist at the clip level, we must also produce AI scores at the clip level.  

> If we were to collapse both clips into one AI call and output a single aggregate score, we would not be able to directly compare AI outputs to the existing human scores on a per‐clip basis.

### 2.2. Avoid Information‐Asymmetry Bias
- **Human Knowledge vs. AI Knowledge**
  Although humans saw both clips (and could draw on context across them), they still recorded _separate_ scores for Clip 1 and Clip 2. If the AI were to see both clips and output a single score, it would have an advantage in “blending” features across clips in ways humans do not when writing down clip‐level ratings.  
- **Maintaining Fair Comparison**  
  By requiring the AI to emit two evaluations (one for each clip), we ensure that:
  1. The evaluation granularity matches human labels.
  2. The AI cannot conflate errors or strengths across Clip 1 and Clip 2 into one blended judgment.

### 2.3. Diagnostic Value
- **Segment‐Level Insights**  
  Teachers often vary in performance between discrete segments. If we collapse clips into one evaluation, we lose the ability to identify which clip drove a particular discrepancy between AI and human scores. Keeping clips separate preserves diagnostic granularity (e.g., “The AI underestimates clarity in Clip 1 but aligns well in Clip 2”).  
