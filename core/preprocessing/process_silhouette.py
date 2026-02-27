import cv2
import numpy as np
import os
import sys

# --- 1. Bulletproof Dependency Loader ---
def initialize_libraries():
    try:
        import mediapipe as mp
        from ultralytics import YOLO
        return mp, YOLO
    except ImportError:
        print("[!] Missing dependencies. Attempting to fix...")
        import subprocess
        # Force install the most stable legacy-compatible version for your pipeline
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mediapipe==0.10.14", "ultralytics", "opencv-python"])
        import mediapipe as mp
        from ultralytics import YOLO
        return mp, YOLO

mp, YOLO = initialize_libraries()

class PreprocessingPipeline:
    def __init__(self, yolo_model='weights/yolov8n.pt'):
        # Initialize YOLO
        self.detector = YOLO(yolo_model)
        
        # Initialize MediaPipe with a fallback check
        try:
            # Attempt to use the classic solutions API
            from mediapipe.solutions import selfie_segmentation
            self.mp_selfie_segmenter = selfie_segmentation.SelfieSegmentation(model_selection=0)
            self._mp_mode = "legacy"
        except (ImportError, AttributeError):
            # Fallback to the modern Tasks API if solutions is missing
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            print("[Info] Using modern MediaPipe Tasks API")
            self._mp_mode = "tasks"
            # In a real task-based setup, you would load a .tflite model here.
            # For this fix, we stick to the stable 0.10.14 version above.

    def background_removal(self, image):
            """
            Uses GrabCut to extract the silhouette based on the image boundaries.
            Since we already cropped the image to the person, the person 
            occupies most of the frame.
            """
            mask = np.zeros(image.shape[:2], np.uint8)
            
            # Internal temporary arrays required by GrabCut
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            # Define a rectangle that slightly excludes the very edge 
            # (GrabCut assumes everything outside the rect is background)
            h, w = image.shape[:2]
            rect = (2, 2, w-4, h-4) 
            
            # Run GrabCut (5 iterations is usually enough)
            cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            
            # GrabCut mask values: 
            # 0 & 2 = Background, 1 & 3 = Foreground
            # We convert this to a binary mask where 1 is person, 0 is BG
            mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Clean up the mask using morphology (removes small holes)
            kernel = np.ones((5, 5), np.uint8)
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
            
            # Create RGBA for the final result
            res = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            res[:, :, 3] = mask_binary * 255
            
            return res, mask_binary

    # ... (Keep quality_enhancement, normalize, object_detection, etc. as they were) ...
    def quality_enhancement(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl,a,b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def object_detection(self, image):
        results = self.detector(image, verbose=False)
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 0: 
                    return box.xyxy[0].cpu().numpy()
        return None

    def image_cropper(self, image, bbox, margin=30):
        h, w, _ = image.shape
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
        return image[y1:y2, x1:x2]

# --- Execution ---
if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)
    pipeline = PreprocessingPipeline()
    
    raw_img = cv2.imread('data/raw/input.png')
    if raw_img is not None:
        enhanced = pipeline.quality_enhancement(raw_img)
        bbox = pipeline.object_detection(enhanced)
        if bbox is not None:
            cropped = pipeline.image_cropper(enhanced, bbox)
            rgba_out, silhouette = pipeline.background_removal(cropped)
            cv2.imwrite('outputs/3_silhouette.png', silhouette * 255)
            cv2.imwrite('outputs/4_final_rgba.png', rgba_out)
            print("Done! Check 'outputs' folder.")