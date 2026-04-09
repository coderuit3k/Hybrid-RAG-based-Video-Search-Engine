import logging
import torch
from transformers import MarianMTModel, MarianTokenizer
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TranslationService:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.device = settings.translation_device
        self.model_name = settings.translation_model
        self.model = None
        self.tokenizer = None
        self._initialized = True
        logger.info(f"TranslationService initialized (lazy load)")

    def _load_model(self):
        """Load model if not already loaded"""
        if self.model is not None:
            return

        try:
            logger.info(f"Loading translation model: {self.model_name}...")
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model = MarianMTModel.from_pretrained(self.model_name).to(self.device)
                
            self.model.eval()
            logger.info("Translation model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise

    def translate_vi_to_en(self, text: str) -> str:
        """
        Translate Vietnamese text to English.
        Returns original text if translation fails.
        """
        if not text or not text.strip():
            return text

        try:
            self._load_model()
            
            # Prepare input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            with torch.no_grad():
                # Generate translation
                translated_tokens = self.model.generate(**inputs)
                translation = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                
            logger.info(f"Translated: '{text}' -> '{translation}'")
            return translation
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text

translation_service = TranslationService()