import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import threading

# --- 1. Global Variables & Model Loading ---

# Configuration
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_ADAPTER_PATH = "./gcm-lora-4171-2"
model = None
tokenizer = None

# Use a lock to prevent multiple threads from loading the model simultaneously
model_lock = threading.Lock()

def load_model():
    """
    Loads the model and tokenizer into global variables.
    This function should only be called once.
    """
    global model, tokenizer
    with model_lock:
        if model is None:
            print("Loading base model and tokenizer...")

            # 4-bit quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                quantization_config=bnb_config,
                device_map={"": 0}
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
            tokenizer.pad_token = tokenizer.eos_token

            # Load and attach the LoRA adapter
            print(f"Loading LoRA adapter from {LORA_ADAPTER_PATH}...")
            model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
            model.eval()
            print("✅ Model loaded successfully!")

# --- 2. Flask App Definition ---

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    The main prediction endpoint.
    Expects a JSON payload with a "prompt" key.
    """
    if model is None or tokenizer is None:
        return jsonify({"error": "Model is not loaded yet. Please try again in a moment."}), 503

    # Ensure the request is JSON
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_prompt = data.get("prompt")

    # Check for the required 'prompt' key
    if not user_prompt:
        return jsonify({"error": "Missing 'prompt' key in request body"}), 400

    try:
        # Format the prompt using the official Mistral template
        prompt = f"<s>[INST] {user_prompt} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Generate the response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode and parse the response to get only the answer
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("[/INST]")[-1].strip()

        return jsonify({"response": answer})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "Failed to generate a response."}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    if model is None or tokenizer is None:
        return jsonify({"status": "unhealthy", "error": "Model is not loaded yet."}), 503

    return jsonify({"status": "healthy"}), 200

# --- 3. Run the App ---

if __name__ == '__main__':
    # Load the model in a separate thread to avoid blocking the server start
    # This is a simple approach for development
    threading.Thread(target=load_model).start()

    # IMPORTANT: Use debug=False when running with large models.
    # The reloader in debug mode can cause the model to load twice,
    # which will crash your GPU due to insufficient memory.
    app.run(host='0.0.0.0', port=5000, debug=False)