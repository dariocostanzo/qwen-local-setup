# run_qwen.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Function to create cache directory if it doesn't exist
def ensure_cache_dir(cache_dir="./model_cache"):
    """
    Creates a local cache directory for storing model weights if it doesn't exist.
    This helps avoid downloading the model repeatedly.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created cache directory at {cache_dir}")
    return cache_dir

# Set up model configuration
def configure_model(quantize=True, cache_dir="./model_cache"):
    """
    Configures the model loading parameters including quantization options.
    
    Args:
        quantize: Whether to use 4-bit quantization (reduces memory usage)
        cache_dir: Directory to cache the downloaded model
    
    Returns:
        Configuration dictionary for model loading
    """
    model_name = "Qwen/QwQ-32B"
    
    # Base configuration
    config = {
        "pretrained_model_name_or_path": model_name,
        "device_map": "auto",  # Automatically distribute model across available GPUs
        "cache_dir": ensure_cache_dir(cache_dir)
    }
    
    # Add quantization if requested (reduces memory requirements significantly)
    if quantize:
        print("Using 4-bit quantization to reduce memory usage")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",  # Normal Float 4 - usually good quality
            bnb_4bit_use_double_quant=True  # Further reduce memory
        )
        config["quantization_config"] = quantization_config
    else:
        print("Loading model in full precision (requires significant GPU memory)")
        config["torch_dtype"] = torch.bfloat16  # Still use bfloat16 to save some memory
    
    return config

# Load the model and tokenizer
def load_model_and_tokenizer(model_config):
    """
    Loads the model and tokenizer with the specified configuration.
    
    Args:
        model_config: Configuration dictionary for model loading
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["pretrained_model_name_or_path"],
        cache_dir=model_config["cache_dir"]
    )
    
    print("Loading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(**model_config)
    
    return model, tokenizer

# Generate a response to a prompt
def generate_response(model, tokenizer, prompt, max_new_tokens=1024):
    """
    Generates a response to the given prompt using the model.
    
    Args:
        model: The loaded Qwen model
        tokenizer: The tokenizer for the model
        prompt: The user's input prompt
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        The model's response as a string
    """
    # Format the prompt as a chat message
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Apply the chat template to format for the model
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    print("Tokenizing input...")
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    print("Generating response...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,  # Controls randomness (0.0 = deterministic, higher = more random)
        top_p=0.9  # Nucleus sampling parameter
    )
    
    # Extract only the newly generated tokens (not including the input prompt)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # Decode the generated tokens into text
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Main function to run the model
def main():
    """
    Main function to set up and run the Qwen/QwQ-32B model.
    """
    print("Setting up Qwen/QwQ-32B...")
    
    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        print(f"CUDA available. Found {torch.cuda.device_count()} GPU(s).")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        
        # Calculate available GPU memory
        gpu_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(gpu_id)
        total_memory_gb = gpu_properties.total_memory / (1024**3)
        print(f"Total GPU memory: {total_memory_gb:.2f} GB")
        
        # Recommend quantization if memory is limited
        should_quantize = total_memory_gb < 80  # 32B models usually need > 64GB in full precision
        if should_quantize:
            print("Your GPU has limited memory. Quantization will be enabled automatically.")
    else:
        print("CUDA not available. Running on CPU (this will be extremely slow).")
        should_quantize = True  # Always quantize on CPU
    
    # Configure and load the model
    model_config = configure_model(quantize=should_quantize)
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # Interactive mode
    print("\nQwen/QwQ-32B model loaded successfully! Enter prompts below (type 'exit' to quit):")
    while True:
        user_prompt = input("\nYou: ")
        if user_prompt.lower() in ["exit", "quit", "q"]:
            break
            
        try:
            response = generate_response(model, tokenizer, user_prompt)
            print(f"\nQwen: {response}")
        except Exception as e:
            print(f"Error generating response: {e}")
    
    print("Exiting program.")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()