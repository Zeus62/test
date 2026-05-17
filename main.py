import os

# =====================================================================
# HARDWARE & MEMORY CONFIGURATION (Must be set before any imports)
# =====================================================================

# OPTION 1: CPU Fallback (Uncomment to bypass a locked GPU entirely)
# os.environ["TORCH_DEVICE"] = "cpu"

# OPTION 2: GPU Diet (Keep these active to use GPU safely without OOM errors)
os.environ["RECOGNITION_BATCH_SIZE"] = "8"   # Lowers VRAM from ~20GB to ~2GB
os.environ["DETECTOR_BATCH_SIZE"] = "2"      # Lowers VRAM from ~9GB to ~1GB

# Prevent PyTorch from heavily fragmenting memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# =====================================================================

import gc
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import numpy as np
import cv2
from PIL import Image
import torch
from langdetect import detect, LangDetectException
from docx import Document
import pdfplumber
import pdf2image

# Surya OCR components
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor

# Setup logging configuration
logger = logging.getLogger("OCR")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class QualityMetrics:
    blur_score: float 
    noise_score: float 
    contrast_score: float 
    resolution_score: float 
    doc_type: str 

    @property
    def overall(self) -> float:
        return (self.blur_score + self.noise_score + self.contrast_score + self.resolution_score) / 4.0

    def __str__(self):
        return (f"blur={self.blur_score:.2f} noise={self.noise_score:.2f} "
                f"contrast={self.contrast_score:.2f} res={self.resolution_score:.2f} "
                f"→ [{self.doc_type}] overall={self.overall:.2f}")


@dataclass
class OCRBlock:
    page: int
    text: str
    confidence: float
    language: str
    bbox: Optional[list]
    strategy: str


class ImageQualityAnalyzer:
    @staticmethod
    def analyze(img_pil: Image.Image) -> QualityMetrics:
        gray = np.array(img_pil.convert("L"))
        h, w = gray.shape

        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = float(np.clip(lap_var / 500.0, 0.0, 1.0))

        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_std = float(np.std(gray.astype(np.float32) - smooth.astype(np.float32)))
        noise_score = float(np.clip(1.0 - noise_std / 30.0, 0.0, 1.0))

        contrast_score = float(np.clip(np.std(gray) / 80.0, 0.0, 1.0))
        resolution_score = float(np.clip((h * w) / (1200.0 * 1200.0), 0.0, 1.0))

        if blur_score < 0.35 or resolution_score < 0.4:
            doc_type = "printed_lq"
        elif contrast_score < 0.4:
            doc_type = "low_quality"
        else:
            doc_type = "printed_hq"

        return QualityMetrics(
            blur_score=blur_score, noise_score=noise_score,
            contrast_score=contrast_score, resolution_score=resolution_score,
            doc_type=doc_type
        )


class Preprocessors:
    @staticmethod
    def _pil_to_bgr(img: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _np_to_pil(bgr: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _upscale(bgr: np.ndarray, min_dim: int = 1200) -> np.ndarray:
        h, w = bgr.shape[:2]
        if min(h, w) < min_dim:
            scale = min_dim / min(h, w)
            return cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        return bgr

    @staticmethod
    def _unsharp_mask(bgr: np.ndarray, sigma: float = 1.0, strength: float = 0.15) -> np.ndarray:
        blurred = cv2.GaussianBlur(bgr, (0, 0), sigma)
        return cv2.addWeighted(bgr, 1.0 + strength, blurred, -strength, 0)

    @staticmethod
    def _lab_percentile_normalize(bgr: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lo, hi = np.percentile(l, [low_pct, high_pct])
        l_norm = np.clip((l.astype(np.float32) - lo) / (hi - lo + 1e-5) * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([l_norm, a, b]), cv2.COLOR_LAB2BGR)

    @classmethod
    def original(cls, img: Image.Image) -> Image.Image:
        return img.convert("RGB")

    @classmethod
    def gentle(cls, img: Image.Image) -> Image.Image:
        bgr = cls._pil_to_bgr(img)
        bgr = cls._upscale(bgr, min_dim=1200)
        bgr = cls._unsharp_mask(bgr, sigma=1.0, strength=0.15)
        return cls._np_to_pil(bgr)

    @classmethod
    def printed_std(cls, img: Image.Image) -> Image.Image:
        bgr = cls._pil_to_bgr(img)
        bgr = cls._upscale(bgr, min_dim=1400)
        bgr = cls._unsharp_mask(bgr, sigma=1.5, strength=0.30)
        bgr = cls._lab_percentile_normalize(bgr, 2.0, 98.0)
        return cls._np_to_pil(bgr)

    @classmethod
    def low_quality(cls, img: Image.Image) -> Image.Image:
        bgr = cls._pil_to_bgr(img)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        
        # Morphological Top-Hat to sharpen thin Arabic text against blurry backgrounds
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        darkened = cv2.subtract(gray, tophat)
        _, crisp = cv2.threshold(darkened, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
        
        return Image.fromarray(cv2.cvtColor(crisp, cv2.COLOR_GRAY2RGB))

    @classmethod
    def handwriting(cls, img: Image.Image) -> Image.Image:
        bgr = cls._pil_to_bgr(img)
        bgr = cls._upscale(bgr, min_dim=1500)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl, a, b))
        enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return cls._np_to_pil(enhanced_bgr)

    @classmethod
    def deskew(cls, img: Image.Image) -> Optional[Image.Image]:
        bgr = cls._pil_to_bgr(img)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        if lines is None: return None
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1: angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if not angles: return None

        median_angle = float(np.median(angles))
        snapped = round(median_angle / 90) * 90
        skew = median_angle - snapped

        if abs(skew) < 0.5: return None 

        h, w = bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
        rotated = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        logger.info(f" ↻ Deskew applied: {skew:+.1f}°")
        return cls._np_to_pil(rotated)

    @classmethod
    def get_candidates(cls, img: Image.Image, metrics: QualityMetrics) -> List[Tuple[str, Image.Image]]:
        candidates = [("original", cls.original(img))]
        deskewed = cls.deskew(img)
        if deskewed: candidates.append(("deskewed", deskewed))

        if metrics.doc_type == "printed_hq":
            candidates.append(("gentle", cls.gentle(img)))
        elif metrics.doc_type == "printed_lq":
            candidates.extend([("gentle", cls.gentle(img)), ("printed_std", cls.printed_std(img)), ("low_quality", cls.low_quality(img))])
        elif metrics.doc_type == "handwritten":
            candidates.extend([("gentle", cls.gentle(img)), ("handwriting", cls.handwriting(img))])
        else: 
            candidates.extend([("gentle", cls.gentle(img)), ("printed_std", cls.printed_std(img)), ("low_quality", cls.low_quality(img))])
        return candidates


class OCRScorer:
    @staticmethod
    def score(blocks: List[OCRBlock]) -> float:
        if not blocks: return 0.0
        confidences = [b.confidence for b in blocks]
        mean_conf = float(np.mean(confidences))
        total_chars = sum(len(b.text.strip()) for b in blocks)
        if total_chars == 0: return 0.0
        return mean_conf * np.log1p(total_chars)


class ContractDocumentPipeline:
    def __init__(self, min_confidence: float = 0.30, dpi: int = 300):
        logger.info(f"Loading Surya OCR models on device: {os.environ.get('TORCH_DEVICE', 'cuda')}...")
        self.detection = DetectionPredictor()
        try:
            from surya.foundation import FoundationPredictor
            self.foundation = FoundationPredictor()
            self.recognition = RecognitionPredictor(self.foundation)
        except ImportError:
            self.recognition = RecognitionPredictor()

        self.min_conf = min_confidence
        self.dpi = dpi
        self.analyzer = ImageQualityAnalyzer()
        self.scorer = OCRScorer()
        logger.info("✅ Pipeline ready — adaptive Arabic/English OCR.")

    @staticmethod
    def _detect_lang(text: str) -> str:
        if len(text.strip()) < 3: return "short"
        try: return detect(text.strip())
        except LangDetectException: return "unknown"

    @staticmethod
    def _extract_native(file_path: str) -> Optional[str]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".docx":
            doc = Document(file_path)
            lines = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(lines) if lines else None
        return None

    def _ocr_image(self, img: Image.Image, page_idx: int) -> List[OCRBlock]:
        metrics = self.analyzer.analyze(img)
        logger.info(f" Page {page_idx + 1}: {metrics}")
        candidates = Preprocessors.get_candidates(img, metrics)
        logger.info(f" Trying {len(candidates)} strategies: {[name for name, _ in candidates]}")

        best_blocks: List[OCRBlock] = []
        best_score = -1.0
        best_name = "original"

        for name, proc_img in candidates:
            try:
                gc.collect()
                if torch.cuda.is_available() and os.environ.get("TORCH_DEVICE") != "cpu":
                    torch.cuda.empty_cache()

                with torch.no_grad():
                    det_results = self.detection([proc_img])
                    predictions = det_results[0]
                    bboxes = [box.bbox for box in predictions.bboxes]
                    rec_results = self.recognition([proc_img], bboxes=[bboxes])[0]
                
                blocks = []
                for idx, text_line in enumerate(rec_results.text_lines):
                    text = text_line.text
                    confidence = getattr(text_line, "confidence", 1.0)
                    bbox = bboxes[idx] if idx < len(bboxes) else None
                    
                    if confidence >= self.min_conf:
                        blocks.append(OCRBlock(
                            page=page_idx, text=text, confidence=confidence,
                            language=self._detect_lang(text), bbox=bbox, strategy=name
                        ))
                
                score = self.scorer.score(blocks)
                if score > best_score:
                    best_score = score
                    best_blocks = blocks
                    best_name = name

                del det_results, predictions, rec_results
            except Exception as e:
                logger.error(f"Strategy '{name}' evaluation encountered an error: {e}")
                if torch.cuda.is_available() and os.environ.get("TORCH_DEVICE") != "cpu":
                    torch.cuda.empty_cache()
                
        logger.info(f" Selected strategy '{best_name}' for page {page_idx + 1} with score {best_score:.2f}")
        return best_blocks

    def process(self, file_path: str, return_structured: bool = False) -> Union[str, dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        native = self._extract_native(file_path)
        if native and native.strip():
            logger.info("Native text extraction succeeded — skipping heavy OCR computation.")
            blocks = [OCRBlock(page=0, text=ln.strip(), confidence=1.0, language=self._detect_lang(ln), bbox=None, strategy="native") for ln in native.splitlines() if ln.strip()]
            if return_structured: return {"source": "native_extraction", "blocks": [b.__dict__ for b in blocks]}
            return "\n".join(b.text for b in blocks)

        ext = os.path.splitext(file_path)[1].lower()
        all_blocks: List[OCRBlock] = []

        if ext == ".pdf":
            images = pdf2image.convert_from_path(file_path, dpi=self.dpi)
            for page_idx, img in enumerate(images):
                all_blocks.extend(self._ocr_image(img, page_idx))
        else:
            try:
                img = Image.open(file_path)
                all_blocks.extend(self._ocr_image(img, 0))
            except Exception as e:
                raise ValueError(f"Could not load image: {e}")

        # ---------------------------------------------------------
        # CRITICAL FIX: Right-To-Left (RTL) Reading Order Sorter
        # ---------------------------------------------------------
        valid_blocks = [b for b in all_blocks if b.bbox is not None]
        no_box_blocks = [b for b in all_blocks if b.bbox is None]
        
        if valid_blocks:
            # 1. Sort primarily Top-to-Bottom (by Y1 coordinate)
            valid_blocks.sort(key=lambda b: b.bbox[1])
            
            lines = []
            curr_line = []
            curr_y_mid = (valid_blocks[0].bbox[1] + valid_blocks[0].bbox[3]) / 2
            
            for b in valid_blocks:
                y_mid = (b.bbox[1] + b.bbox[3]) / 2
                h = b.bbox[3] - b.bbox[1]
                
                # If the box is roughly on the same horizontal plane, group it into the line
                if abs(y_mid - curr_y_mid) < (h / 1.5):
                    curr_line.append(b)
                else:
                    # 2. Sort the completed line Right-to-Left (by X2 coordinate descending)
                    curr_line.sort(key=lambda b: b.bbox[2], reverse=True)
                    lines.extend(curr_line)
                    curr_line = [b]
                    curr_y_mid = y_mid
                    
            if curr_line:
                curr_line.sort(key=lambda b: b.bbox[2], reverse=True)
                lines.extend(curr_line)
                
            all_blocks = no_box_blocks + lines
        # ---------------------------------------------------------

        if return_structured:
            return {"source": "ocr_pipeline", "blocks": [b.__dict__ for b in all_blocks]}
        return "\n".join(b.text.strip() for b in all_blocks if b.text.strip())


_pipeline: Optional[ContractDocumentPipeline] = None

def get_pipeline(min_confidence: float = 0.30, dpi: int = 300) -> ContractDocumentPipeline:
    global _pipeline
    if _pipeline is None: _pipeline = ContractDocumentPipeline(min_confidence=min_confidence, dpi=dpi)
    return _pipeline

def process_contract(file_path: str, return_structured: bool = False, min_confidence: float = 0.30, dpi: int = 300) -> Union[str, dict]:
    return get_pipeline(min_confidence, dpi).process(file_path, return_structured)

if __name__ == "__main__":
    pass