from transformers import AutoModelForCausalLM
from PIL import Image
from config import MODEL_NAME, MODEL_REVISION, DEVICE

class ModelHandler:
    def __init__(self):
        self.model = None
        
    def load_model(self):
        
        if self.model is not None:
            return
            
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-01-09",
            trust_remote_code=True,
        ).to(DEVICE)
        
    def detect_objects(self, image: Image.Image, target_object: str) -> list:
 
        if self.model is None:
            self.load_model()
            
        return self.model.detect(image, target_object)["objects"] 