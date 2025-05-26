# SwiftScore Montesa

Montesa is the product of a collaboration between SwiftScore and the World Bank aimed at aligning AI-generated teacher performance evaluations with human judgments using classroom audio transcripts.

## Project Overview

Montesa leverages artificial intelligence to analyze classroom audio transcripts and produce evaluative feedback aligned with the World Bank's established Teach framework. The project aims to develop a scalable and privacy-preserving teacher evaluation system that closely aligns with human expert judgments.

### Key Features

- **Audio-Based Evaluation**: Processes classroom audio transcripts to produce evaluations, making it accessible for schools worldwide without requiring video infrastructure
- **Privacy Preservation**: Utilizes differential privacy techniques to protect teacher and student identities
- **Teach Framework Alignment**: Evaluates teaching quality based on the World Bank's established Teach framework dimensions
- **Synthetic Data Generation**: Creates high-quality synthetic training data to enhance model performance without compromising privacy

## Repository Structure

- **`/documentation`**: Project documentation and planning materials
- **`/training`**: Training scripts and configuration for model fine-tuning
- **`/eval`**: Evaluation scripts and metrics for assessing model performance
- **`/synthetic-data-generation`**: Scripts for generating synthetic classroom observations and evaluations
- **`/data`**: Input and output data files for model training and evaluation

## Technical Approach

Montesa follows a four-stage process:

1. **Data Cleaning and EDA**: 
   - Standardize and clean classroom observation data
   - Transcribe classroom audio using speech-to-text models
   - Perform exploratory data analysis to understand data characteristics

2. **Synthetic Data Generation**:
   - Use Gemini 2.0 Flash as a "teacher" model to generate high-quality synthetic evaluations
   - Apply differential privacy techniques to ensure privacy preservation
   - Create a large corpus of (transcript, evaluation) pairs for training

3. **Supervised Fine-Tuning**:
   - Utilize Quantized Low-Rank Adaptation (QLoRA) for efficient model fine-tuning
   - Train multiple model variants to compare performance with different data mixtures
   - Maintain 4-bit quantization for optimized inference

4. **Model Evaluation**:
   - Assess alignment with human evaluations using inter-rater reliability metrics
   - Compare with human-to-human reliability benchmarks
   - Conduct qualitative error analysis and visualization

## Usage

### Synthetic Data Generation

```bash
python -m synthetic-data-generation.synthetic_data_generator --count 10 --workers 4
```

### Model Training

```bash
python training/train.py --data-dir data/ --output-dir trained_llm --model google/flan-t5-base --chat-template chatml
```

## Privacy and Ethics

Montesa prioritizes:
- **Privacy Preservation**: Using differential privacy to protect sensitive educational data
- **Human-in-the-Loop Design**: Ensuring AI enhances rather than replaces human judgment
- **Ethical AI Principles**: Developing AI that respects teacher autonomy and professional norms

## Collaboration Partners

- **SwiftScore**: AI company specializing in teacher performance evaluation
- **World Bank**: Provider of the Teach framework and educational expertise

## License

The fine-tuning code and methodology in this repository are open-source, while the resulting trained model and SwiftScore proprietary components remain the intellectual property of SwiftScore.