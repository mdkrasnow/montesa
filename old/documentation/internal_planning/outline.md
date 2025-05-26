# Sources

[Outline](https://www.notion.so/Outline-1d075584fbef80768da7fcf90c3c449e?pvs=21)

[Proposal](https://www.notion.so/Proposal-1d075584fbef80e39ff0f31cb88ca780?pvs=21)

[SwiftScore x World Bank Proposal](https://www.notion.so/SwiftScore-x-World-Bank-Proposal-1d575584fbef8018948deb35a708edd0?pvs=21)

[[Final] SwiftScore x World Bank Proposal](https://www.notion.so/Final-SwiftScore-x-World-Bank-Proposal-1d575584fbef800ebd8ad10bf66ccfdc?pvs=21)

# Timeline

Minimum: 9 days

Maximum: 27 days

**Stage 1: Data Cleaning and EDA**

- Minimum: 1 day
- Maximum: 3 days

**Stage 2: Synthetic Data Generation**

- Minimum: 4 days
- Maximum: 12 days

**Stage 3: Supervised Fine-tuning**

- Minimum: 2 days
- Maximum: 6 days

**Stage 4: Model Evaluation**

- Minimum: 2 days
- Maximum: 6 days

Just as some background, I want you to understand that swiftscore is an AI evaluation company which creates evaluations for teachers, which replaces having to have a human evaluator actually review the teacher's progress. And we're investigating if the AI evaluations are actually reliable and score similarly to the human's evaluator. So the AI is capable of producing similar responses to the human's. This is what we're interested in, so swiftscore is not necessarily interested in cutting out humans from the picture, but it is really the idea that we're trying to test if an AI trained on data is able to perform similarly as a human and demonstrate that reliably. Then, based on the data and this findings, we would then, with the World Bank, write a paper demonstrating the findings and publish it. We would also open source any code related to fine tuning this and any other algorithms that we use. However, the property would still be of swiftscore. Swiftscore owns the code and swiftscore would own the product at the end of it.

# Stage 1: Data Cleaning and EDA ( 1-2 days)

timeline:

- min: 1 day
- max: 3 days

Okay, so essentially for data cleaning and EDA, what we're going to do is we're going to take whatever format that the World Bank provides the data with and then analyze how we need to clean it so that it is properly in like a CSV format so that we can actually look at the data and see what we're working with, note things like number of examples that are given to us, number of like data points. And then we want to see the format of the evaluations and if this works in a compatible format for eval, we need to then if the evaluation format is compatible with eval, can we easily translate it, we will probably use an llm to translate between the two as data cleaning, if not we're going to have to inform the World Bank and let them know that we're going to be able to do that. So let them know the situation with cleaning the data and reformatting it in the same way. Part of this is going to be so we know that the transcriptions are in like we need to create the transcriptions from audio data of what the classroom is saying, what's going on so we're going to use the speech to text model which will transcribe the student class. So we're going to use the student classroom teacher experience information, then we will format this into the CSV as the input, the low inference notes column, and then we can do some simple data visualizations so on basic data statistics of kind of what we're working with and finish out the planning of what we need to do if there are any changes we need to make.

- Clean data that is provided from the world bank
- Take audio data format content and transcribe using a STT model
- Format it into CSV for analysis
- Exploratory Data Analysis
    - Plan out the specifics of how we will proceed with the project based on
        - Number of real examples
        - Quality of evaluations
        - Expected format of evaluations

# Stage 2: Synthetic Data Generation (days to weeks)

Timeline:

- min: 4 days
- max: 12 days

Sources:

https://www.perplexity.ai/search/i-have-a-machine-learning-prob-Bmzdt7.eReqpDDpFANl.lg

Okay, so essentially for synthetic data generation, we are going to need to take a teacher model, so a smart llm like Gemini 2.0 flash, and use it to create pairs of data. So input pairs is an input paired with an output, so we need the input to reflect what an actual transcription from one of these classrooms might look like, and then create a reasonable input. Then we're from this input and a set of input-output pairs create an output of the evaluation in the necessary format. This we will have to do this and repeat this is that a generation for to generate as many data points as we previously determined and agree upon with the World Bank given computing and time constraints. Then we'll have to reformat the data so that it is in the proper format for the training pipeline for which we decide to use. In this stage, if agreed upon with the World Bank, we may implement a differential privacy algorithm when generating the synthetic data, we likely want to do this and I would push for it at a higher cost because differential privacy is very important and it is previously shown that you can generate synthetic data for llm's and it's useful for later swiftscore processes because we want to justify training on evaluator or school data without compromising their privacy or security so I think this would be very beneficial research for swiftscore and for the world bank to show that educational data can be used effectively and ethically to improve ai outcomes without compromising personal information.

- Determine the most appropriate synthetic data generation method based on the format of the data
    - Teacher model choice: Gemini 2.0 Flash
        - why? smart, fast, cheap, long context window, etc.
    - Student model choice: Gemma 3 27B, Scout
        - why? small focused, open weights, good DP resources
- Create synthetic data using cutting-edge ML theory
    - Differential Privacy Algorithms
- Two step SDG pipeline
    - Need to generate context aware inputs (transcripts)
    - Based on the specific transcripts, we need to generate evaluations in the same format
- Reformat data into necessary format for training

# Stage 3: Supervised Fine-tuning (days to week)

timeline

- min: 2 days
- max: 6 days

Third step is going to be supervised fine-tuning of the learner model. So this is going to be a small model which is open weights and we're going to choose a small model because these models are generally accept fine-tuning well and can perform on par with large LLMs without the compute overhead after being fine-tuned on a specific task. So we are going to perhaps train a few different models depending on what we agree with at the World Bank, probably one that is trained exclusively on synthetic data, one that is trained on a mixture of synthetic and real data and a final one which is not trained on anything at all which is really just the base model and we're going to see how it performs later on in the model evaluations section. The supervised fine-tuning that we will probably do will be a QLORA so it will be a quantized low-rank adaptation of the model parameters so we're going to be not modifying all the model parameters, simply just adding or modifying a single kind of layer which will allow us to tune the model. So we're going to be going to see how it performs later on in the model evaluations section. We will quantize the model weights because it's been previously demonstrated that quantization does not significantly impact performance or intelligence while significantly reducing the training speed, inference speed, etc.We will likely use a four-bit quantization. 

- Determine which parameters we are training
    - Quantized Low rank adaptation: QLoRA
- Determine loss function(s)
- Train models on synthetic data (and a real data mixture)
    - * determine what models will be trained

# Stage 4: Model Evaluation (days to week)

timeline:

- min: 2 days
- max: 6 days

Finally, once we have the tune models, we will run inference evaluations on the models. So ideally we will have already allocated a test set of the real data and perhaps also generate some unseen synthetic data to evaluate the model. We will also agree with the World Bank on what the evaluation metrics are. So for example, score accuracy, if the score that is given by the AI matches the score that is given by the human, The semantic similarity between evaluations that are generated by the AI and versus the human, we may also use some kind of other scoring like rogue or blue as metrics for determining how similar the evaluations are. If possible, we would like to evaluate these AI versus human evaluations and for actual evaluators to decide which ones they like better, which ones they prefer. So we can do this in a kind of ELO, kind of choosing way where evaluators are able to decide which output they prefer and we can then see if the AI outperforms the humans. However, this would be kind of an add-on and would require actual time because we would need human evaluators to review the data in real time. We would then create model visualizations for displaying the metrics that we tracked to show the benefit. Ideally, we could also test other models to see how well they perform on this kind of benchmark, so we could test smarter models like Gemini, which are untuned just to see how well our performance and tuning actually improves models. So we take the base model, which we fine-tuned, take each fine-tuned example of that model, and then try some other models just for comparison.

- Determine the model evaluation metrics
- Evaluate the models based on the real data
- Create visualizations of model performance
    - Overall model alignment with human evaluators
    - Improvement over the base model and inter-model differences

---

ignore for now

## Add ons — extra price

- Increased privacy algorithms
    - Using cutting edge differential privacy to mathematiclaly guaruntee that we are protecting user information
    - Using Piirhana to clean data prior to SDG
- More models
    - Train more models on the data
- More data
    - data is costly to generate
        - how much do we want to generate?