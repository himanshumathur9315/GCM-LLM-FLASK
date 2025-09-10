import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import threading
import traceback

# --- 1. Global Variables & Model State ---

# Configuration
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_ADAPTER_PATH = "./gcm-lora-4171-2"

# --- State Variables ---
model = None
tokenizer = None
model_loading_error = None  # To capture any errors during model loading

# Use a lock to prevent multiple threads from accessing the model simultaneously
model_lock = threading.Lock()

def load_model():
    """
    Loads the base model and LoRA adapter.
    Captures and stores any exceptions that occur during the process.
    """
    global model, tokenizer, model_loading_error
    try:
        with model_lock:
            # Don't try to load if it's already loaded or an error occurred
            if model is not None or model_loading_error is not None:
                return

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

    except Exception as e:
        # Capture the error for the health check endpoint
        model_loading_error = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(f"❌ An error occurred during model loading: {model_loading_error}")

# --- 2. Flask App Definition ---

app = Flask(__name__)

@app.route('/')
def index():
    """
    A simple root endpoint to confirm the server is running and show available routes.
    """
    return jsonify({
        "message": "Welcome to the Mistral Model API!",
        "endpoints": {
            "health_check": {
                "path": "/health",
                "method": "GET"
            },
            "prediction": {
                "path": "/predict",
                "method": "POST",
                "body": {"prompt": "Your text here"}
            }
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    The main prediction endpoint.
    Expects a JSON payload with a "prompt" key.
    """
    # Check for model loading errors first
    if model_loading_error:
        return jsonify({"error": "Model is unavailable due to a loading error.", "details": model_loading_error}), 503

    if model is None or tokenizer is None:
        return jsonify({"error": "Model is not loaded yet. Please try again in a moment."}), 503

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_prompt = data.get("prompt")

    if not user_prompt:
        return jsonify({"error": "Missing 'prompt' key in request body"}), 400

    try:
        prompt = f"<s>[INST] {user_prompt} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                eos_token_id=tokenizer.eos_token_id
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("[/INST]")[-1].strip()

        return jsonify({"response": answer})

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({"error": "Failed to generate a response."}), 500

@app.route('/health')
def health_check():
    """
    Health check endpoint that provides the current status of the model.
    """
    # 1. Check if an error occurred during loading
    if model_loading_error:
        return jsonify({
            "status": "unhealthy",
            "reason": "An error occurred during model loading.",
            "details": model_loading_error
        }), 503

    # 2. Check if the model is still loading
    if model is None or tokenizer is None:
        return jsonify({
            "status": "unhealthy",
            "reason": "Model is still loading. Please wait."
        }), 503

    # 3. If loaded, confirm it's responsive
    try:
        tokenizer("test", return_tensors="pt")
        return jsonify({
            "status": "healthy",
            "details": {
                "model_status": "loaded",
                "tokenizer_status": "loaded",
                "device": str(next(model.parameters()).device)
            }
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "reason": f"Model verification failed after loading: {str(e)}"
        }), 503

# --- 3. Run the App ---

if __name__ == '__main__':
    # Load the model in a background thread to avoid blocking the server start
    threading.Thread(target=load_model).start()

    # IMPORTANT: Use debug=False when running with large models, as the reloader
    # in debug mode can cause a crash by loading the model twice.
    app.run(host='0.0.0.0', port=5000, debug=False)
