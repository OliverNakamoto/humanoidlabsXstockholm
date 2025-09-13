import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image

class FlyVLM:
    def __init__(self, model_id="google/paligemma-3b-mix-224"):
        """
        Initializes and loads the quantized PaliGemma model.
        """
        print(f"Loading VLM: {model_id}. This may take a while on the first run...")
        
        # Load the model with 4-bit quantization to save VRAM
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.prompt = "is there a fly on the racket?"
        print("VLM loaded successfully.")

    def check_for_fly(self, cv2_image) -> bool:
        """
        Takes an OpenCV image, asks the model if a fly is present, and returns True or False.
        """
        try:
            # Convert OpenCV image (BGR) to PIL image (RGB)
            image = Image.fromarray(cv2_image[:, :, ::-1])
            
            # Prepare the inputs for the model
            inputs = self.processor(text=self.prompt, images=image, return_tensors="pt").to(self.model.device)
            
            # Generate the response
            output = self.model.generate(**inputs, max_new_tokens=10)
            
            # Decode the response and check for a "yes"
            response_text = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Simple check to see if the model answered affirmatively
            # The model usually outputs "yes" or "no" as the first word.
            if "yes" in response_text.lower():
                return True
            return False
            
        except Exception as e:
            print(f"Error during VLM inference: {e}")
            return False