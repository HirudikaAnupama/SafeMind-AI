import torch
import os
import numpy as np
from collections import Counter
from flask import Flask, request, jsonify, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig
import bert_score

# Set Model Directory
MODEL_DIR = r"C:\Users\LENOVO\Desktop\Projects\SafeMind AI\t5_finetuned_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get Checkpoints
CHECKPOINT_PATHS = [
    os.path.join(MODEL_DIR, f"checkpoint-{i}") for i in range(1, 4)
    if os.path.exists(os.path.join(MODEL_DIR, f"checkpoint-{i}"))
]

if not CHECKPOINT_PATHS:
    print("No valid checkpoints found in the specified directory.")
else:
    print(f"Found Checkpoints: {CHECKPOINT_PATHS}")

# Load Tokenizer
try:
    tokenizer = T5Tokenizer.from_pretrained(CHECKPOINT_PATHS[-1]) if CHECKPOINT_PATHS else None
except Exception as e:
    print(f"Error loading tokenizer: {str(e)}")
    tokenizer = None

# Load Models
models = []
for path in CHECKPOINT_PATHS:
    try:
        config = AutoConfig.from_pretrained(path)
        model = T5ForConditionalGeneration.from_pretrained(
            path, config=config, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device).eval()
        models.append(model)
    except Exception as e:
        print(f"Warning: Error loading model from {path}: {str(e)}")

if not models:
    print("No models loaded. Please check the checkpoint paths.")

# Define Expanded Mental Health Keywords for Reinforcement
MENTAL_HEALTH_KEYWORDS = {
    "self-care", "therapy", "support", "healing", "emotions", "mindfulness", "relaxation", "breathing", "coping", "balance",
    "well-being", "positivity", "resilience", "calmness", "serenity", "meditation", "acceptance", "awareness", "inner peace",
    "psychotherapy", "mental fitness", "counseling", "self-compassion", "empathy", "gratitude", "self-reflection", "stress relief",
    "emotional intelligence", "mood regulation", "holistic health", "decompression", "grounding", "clarity", "personal growth",
    "self-acceptance", "emotional support", "self-improvement", "therapy sessions", "mental clarity", "positivity exercises",
    "coping mechanisms", "adaptive thinking", "self-awareness", "psychological support", "health counseling", "peace of mind",
    "confidence building", "emotional healing", "mindset shift", "mental training", "happiness exercises", "wellness routines",
    "life coaching", "stress management", "behavioral therapy", "cognitive flexibility", "self-motivation", "self-discipline",
    "journaling", "progress tracking", "self-efficacy", "goal setting", "self-empowerment", "inner strength", "mindful breathing",
    "healing practices", "meditative focus", "positive affirmations", "mindful movement", "holistic therapy", "emotional grounding",
    "psychological balance", "mind-body connection", "compassionate communication", "anxiety relief", "mental detox",
    "mindful awareness", "therapy exercises", "psychoeducation", "emotional wisdom", "self-nurturing", "self-growth",
    "personal development", "mindfulness techniques", "stress reduction", "healing journey", "behavioral coaching",
    "mood enhancement", "coping strategies", "happiness coaching", "peaceful mindset", "support network", "relaxation techniques",
    "stress resilience", "mental adaptability", "mindful lifestyle", "emotional processing", "psychological resilience",
    "supportive relationships", "guided therapy", "cognitive wellness", "positive psychology", "reframing negative thoughts",
    "transformational healing", "emotional adaptability", "self-kindness", "mental energy", "healthy boundaries",
    "supportive community", "relaxation response", "trauma recovery", "grief counseling", "life transitions",
    "emotional validation", "thought restructuring", "growth mindset", "self-help strategies", "emotional comfort",
    "body-mind synchronization", "healing rituals", "psychological well-being", "mental harmony", "compassion training",
    "empowerment coaching", "behavioral adaptation", "therapeutic communication", "inner transformation",
    "restorative self-care", "psychosocial wellness", "emotional safety", "self-soothing techniques", "mindfulness therapy",
    "emotional freedom", "peaceful reflection", "mindful eating", "compassion-based therapy", "self-renewal"
}

# Function for Enhanced Ensemble Prediction
def ensemble_generate(prompt, num_return_sequences=3):
    if not isinstance(prompt, str) or not prompt.strip():
        return "Invalid input: Prompt must be a non-empty string."

    if not tokenizer or not models:
        return "Model or tokenizer not loaded. Unable to generate response."

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    responses_with_scores = []

    for i, model in enumerate(models):
        temperature = max(0.3, 1 - (0.1 * len(models)))  # Adaptive temperature
        try:
            outputs = model.generate(
                input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=300,
                num_beams=8,
                early_stopping=True,
                repetition_penalty=3.0,
                length_penalty=2.0,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=num_return_sequences
            )

            logits = torch.stack(outputs.scores).mean().item() if outputs.scores else 0
            model_weight = (i + 1) * 1.5

            for output in outputs.sequences:
                response = tokenizer.decode(output, skip_special_tokens=True).strip()
                confidence_score = logits * model_weight

                keyword_count = sum(1 for word in MENTAL_HEALTH_KEYWORDS if word in response.lower())
                confidence_score += keyword_count * 2

                responses_with_scores.append((response, confidence_score))

        except Exception as e:
            print(f"Warning: Error generating response with model {i}: {str(e)}")

    response_scores = Counter()
    for response, score in responses_with_scores:
        response_scores[response] += score

    top_responses = [resp for resp, _ in response_scores.most_common(5)]
    if not top_responses:
        return "I'm sorry, but I couldn't generate a response."

    try:
        P, R, F1 = bert_score.score(
            top_responses, [prompt] * len(top_responses), lang="en", model_type="bert-base-uncased"
        )
        best_response = top_responses[np.argmax(F1.cpu().numpy())]
    except Exception as e:
        print(f"Warning: Error in BERTScore computation: {str(e)}")
        best_response = top_responses[0]

    best_response = ". ".join(sent.capitalize() for sent in best_response.split(". "))
    return best_response


# Initialize Flask App
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        prompt = data.get("text", "").strip()

        if not prompt:
            return jsonify({"error": "No text provided"}), 400

        print(f"Received Prompt: {prompt}")
        response = ensemble_generate(prompt)
        print(f"Generated Response: {response}")
        return jsonify({"response": response})

    except Exception as e:
        print(f"Error in Response Generation: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
