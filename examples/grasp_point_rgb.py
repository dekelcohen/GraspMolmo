from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torch

def load_model(model_name="allenai/GraspMolmo"):
    """
    Load the processor and model for GraspMolmo (or similar multimodal grasping model).
    Returns (processor, model).
    """
    processor = AutoProcessor.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    return processor, model

def grasp_inference(processor, model, image_path: str, task: str) -> str:
    """
    Given a processor, model, image path, and a high-level task string,
    returns the model’s predicted grasp location text.
    """
    img = Image.open(image_path).convert("RGB")
    prompt = f"Point to where I should grasp to accomplish the following task: {task}"
    inputs = processor.process(images=img, text=prompt, return_tensors="pt")
    # Move to model device and batch dimension
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=256, stop_strings="<|endoftext|>")
    )
    # The generated tokens beyond the input length
    generated_tokens = output[0, inputs["input_ids"].size(1):]
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )
    return generated_text

def main():
    processor, model = load_model()
    image_path = "images/yellow_duck_with_so_101_arm.png"
    task = "Pick up the yellow duck."
    result = grasp_inference(processor, model, image_path, task)
    print("Grasp prediction:", result)

if __name__ == "__main__":
    main()
