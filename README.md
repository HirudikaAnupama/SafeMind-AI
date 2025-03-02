# SafeMind AI - Mental Health Chatbot

## Overview
SafeMind AI is a mental health chatbot designed to provide empathetic and supportive responses to users experiencing emotional distress. The chatbot leverages a fine-tuned T5 model, trained on a diverse dataset of over 70,000 mental health-related interactions, to generate meaningful and actionable responses.

## Features
- **Natural Language Understanding**: Uses a fine-tuned T5 model for contextual comprehension.
- **Ensemble Approach**: Combines multiple model checkpoints to enhance response quality.
- **Keyword Reinforcement**: Prioritizes responses containing mental health-related keywords.
- **Flask Backend**: API server for model inference and chatbot interactions.
- **User Interface**: A front-end integrated with the backend for seamless user experience.

## Data Preprocessing
- **Cleaning & Normalization**: Applied text cleaning techniques to remove noise, unwanted characters, and shorthand words.
- **HTML Parsing**: Extracted text content from HTML-formatted responses.
- **Deduplication & Oversampling**: Removed duplicates and balanced the dataset using random oversampling.
- **Final Dataset**: Processed over 70,000+ mental health-related interactions.

## Model Training
- **Dataset**: Used a curated dataset from various sources, cleaned and preprocessed.
- **Tokenization**: Applied T5 tokenizer with padding and truncation.
- **Fine-Tuning**: Trained T5-small using PyTorch and Hugging Face's Trainer API.
- **Training Parameters**:
  - Learning Rate: `3e-4`
  - Batch Size: `8`
  - Epochs: `5`
  - Gradient Accumulation: `4`
  - Early Stopping & Checkpointing Enabled

## Ensemble Inference
- **Multiple Checkpoints**: Utilized the best three checkpoints for ensemble generation.
- **Keyword Boosting**: Enhanced response selection based on mental health-related keywords.
- **BERTScore Evaluation**: Applied similarity filtering to select the most relevant response.

## Backend (Flask API)
- **Model Hosting**: Serves the fine-tuned T5 model for inference.
- **Request Handling**: Accepts user queries and returns generated responses.
- **Deployment-Ready**: Designed for cloud deployment.

## Frontend
- **User Interface**: Interactive chatbot UI for engaging conversations.
- **Seamless Communication**: Integrated with Flask backend for real-time response generation.

## Installation & Setup
```sh
# Clone the repository
git clone https://github.com/HirudikaAnupama/SafeMind-AI
cd SafeMindAI

# Install dependencies
pip install -r requirements.txt

# Run the Flask backend
python app.py

# (Optional) Deploy the frontend
# Setup frontend framework of choice
```

## Example Usage
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5_finetuned_model")
model = T5ForConditionalGeneration.from_pretrained("t5_finetuned_model")

def generate_answer(question):
    input_text = f"Provide an empathetic response to: {question}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids)
    return tokenizer.decode(output[0], skip_special_tokens=True)

question = "I feel anxious all the time."
print(generate_answer(question))
```

## Future Improvements
- Expand dataset with more diverse interactions.
- Enhance model performance using larger architectures.
- Implement real-time chat capabilities.

## License
This project is licensed under the MIT License.

---
Developed by **Hirudia Anupama** - [GitHub](https://github.com/HirudikaAnupama)

<br>

![Sample Chat](https://github.com/user-attachments/assets/bc599715-23f4-425a-beaf-10924d8e718f)




