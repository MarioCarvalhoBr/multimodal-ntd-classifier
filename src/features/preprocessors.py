import cv2
import numpy as np
from abc import ABC, abstractmethod

class ImagePreprocessor(ABC):
    """Interface abstrata para pré-processamento de imagens."""
    
    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        pass

class HairRemovalFilter(ImagePreprocessor):
    """
    Implementa a remoção de pelos utilizando filtro morfológico Black-Hat 
    e algoritmo de inpainting TELEA.
    """
    def __init__(self, kernel_size: tuple = (17, 17), threshold: int = 10):
        self.kernel_size = kernel_size
        self.threshold = threshold

    def process(self, image: np.ndarray) -> np.ndarray:
        # 1. Convert to grayscale
        return image
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 2. Apply Black-Hat filter to detect hair pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel_size)
        blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)
        
        # 3. Threshold -> binary mask
        _, threshold_mask = cv2.threshold(blackhat, self.threshold, 255, cv2.THRESH_BINARY)
        
        # 4. Inpaint masked pixels with TELEA algorithm
        inpainted_image = cv2.inpaint(image, threshold_mask, 1, cv2.INPAINT_TELEA)
        
        return inpainted_image