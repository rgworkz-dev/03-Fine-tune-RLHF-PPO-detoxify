# Fine-Tune FLAN-T5 with Reinforcement Learning (PPO) and PEFT to Generate Less-Toxic Summaries

This repository contains a Jupyter notebook that demonstrates how to fine-tune a FLAN-T5 model to generate less toxic content with Meta AI's hate speech reward model. The reward model is a binary classifier that predicts either "not hate" or "hate" for the given text. We will use Proximal Policy Optimization (PPO) to fine-tune and reduce the model's toxicity.

## Overview

The notebook covers the following key steps:

1. Loading and preprocessing the DialogSum dataset
2. Setting up the FLAN-T5 model with PEFT (Parameter-Efficient Fine-Tuning)  
3. Preparing the toxicity reward model and evaluator
4. Fine-tuning the model using PPO to reduce toxicity
5. Evaluating the model quantitatively and qualitatively

## Key Components

- FLAN-T5: Base language model for summarization
- PEFT: For efficient fine-tuning of the base model
- Meta AI's RoBERTa-based hate speech model: Used as the reward model
- PPO (Proximal Policy Optimization): RL algorithm for fine-tuning
- Hugging Face Transformers & Datasets libraries

## Requirements

- Python 3.7 or above
- Jupyter Notebook
- Hugging Face Transformers
- Datasets
- PEFT (Parameter-Efficient Fine-Tuning)

You can install the required dependencies with the following command:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/rgworkz-dev/03-Fine-tune-RLHF-PPO-detoxify.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:
   ```bash
   jupyter notebook fine_tune_model_to_detoxify_summaries.ipynb
   ```

4. Run the cells in the notebook sequentially to fine-tune the model and evaluate its performance.

## Results

The notebook demonstrates a significant reduction in the toxicity of generated summaries after fine-tuning, as measured by:

1. Quantitative improvement in toxicity scores
2. Qualitative comparison of summaries before and after detoxification

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for providing the pre-trained FLAN-T5 model and tools.
- Meta AI
- The open-source community for datasets and support.