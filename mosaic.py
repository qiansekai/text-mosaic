# -*- coding: utf-8 -*-
"""
文字打码工具 - PaddleOCR 3.0 字符级精确打码
支持长图处理，性能优化版
"""

import cv2
import numpy as np
from PIL import Image, ImageGrab
import argparse
import os
import tempfile
import warnings

warnings.filterwarnings('ignore')
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCR


class TextMosaic:
    _ocr_instance = None  # 单例，避免重复加载模型
    
    def __init__(self, use_fast_model=True):
        """
        初始化 OCR
        use_fast_model: True 使用轻量模型(快), False 使用服务器模型(准)
        """
        if TextMosaic._ocr_instance is None:
            if use_fast_model:
                # 轻量模型 + mkldnn 加速
                TextMosaic._ocr_instance = PaddleOCR(
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device="cpu",
                    enable_mkldnn=True,
                )
            else:
                # 服务器模型，更准确但更慢
                TextMosaic._ocr_instance = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device="cpu",
                    enable_mkldnn=True,
                )
        self.ocr = TextMosaic._ocr_instance
    
    def apply_mosaic(self, image, box, block_size=10):
        """对指定区域应用马赛克效果"""
        x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        h, w = image.shape[:2]
        x_min, x_max = max(0, x_min), min(w, x_max)
        y_min, y_max = max(0, y_min), min(h, y_max)
        
        if x_max <= x_min or y_max <= y_min:
            return image
        
        roi = image[y_min:y_max, x_min:x_max]
        roi_h, roi_w = roi.shape[:2]
        bs = max(1, min(block_size, roi_h, roi_w))
        
        small = cv2.resize(roi, (max(1, roi_w // bs), max(1, roi_h // bs)))
        mosaic = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        image[y_min:y_max, x_min:x_max] = mosaic
        return image
    
    def process_image(self, image_path, target_texts, output_path=None, block_size=10):
        """处理图片，使用字符级边界框精确打码"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        h, w = image.shape[:2]
        print(f"图片尺寸: {w}x{h}")
        
        # OCR识别，启用字符级边界框
        results = self.ocr.predict(image_path, return_word_box=True)
        
        mosaic_count = 0
        target_set = set(target_texts)  # 用 set 加速查找
        
        for res in results:
            rec_texts = res.get('rec_texts', [])
            text_words = res.get('text_word', [])
            text_word_boxes = res.get('text_word_boxes', [])
            
            if not text_words or not text_word_boxes:
                continue
            
            for line_text, words, word_boxes in zip(rec_texts, text_words, text_word_boxes):
                if not words or word_boxes is None:
                    continue
                
                # 快速检查：这行是否包含任何目标文字
                if not any(t in line_text for t in target_set):
                    continue
                
                word_boxes_list = word_boxes.tolist() if hasattr(word_boxes, 'tolist') else word_boxes
                combined = ''.join(words)
                
                for target in target_set:
                    if target not in line_text:
                        continue
                    
                    start = 0
                    while True:
                        pos = combined.find(target, start)
                        if pos == -1:
                            break
                        
                        char_count = 0
                        target_end = pos + len(target)
                        
                        for word_idx, word in enumerate(words):
                            word_start = char_count
                            word_end = char_count + len(word)
                            
                            if word_end > pos and word_start < target_end:
                                if word_idx < len(word_boxes_list):
                                    self.apply_mosaic(image, word_boxes_list[word_idx], block_size)
                                    mosaic_count += 1
                            
                            char_count = word_end
                        
                        start = pos + 1
        
        print(f"共打码 {mosaic_count} 处")
        
        if output_path is None:
            name, ext = os.path.splitext(image_path)
            output_path = f"{name}_mosaic{ext}"
        
        cv2.imwrite(output_path, image)
        print(f"已保存到: {output_path}")
        return output_path


def get_clipboard_image():
    """从剪贴板获取图片"""
    try:
        img = ImageGrab.grabclipboard()
        if img is None:
            print("剪贴板中没有图片")
            return None
        if isinstance(img, list):
            return img[0] if img else None
        temp_path = os.path.join(tempfile.gettempdir(), "clipboard_image.png")
        img.save(temp_path)
        print("已从剪贴板获取图片")
        return temp_path
    except Exception as e:
        print(f"获取剪贴板图片失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='对图片中的指定文字进行打码')
    parser.add_argument('image', nargs='?', help='输入图片路径')
    parser.add_argument('-t', '--text', nargs='+', required=True, help='要打码的文字')
    parser.add_argument('-o', '--output', help='输出图片路径')
    parser.add_argument('-b', '--block-size', type=int, default=10, help='马赛克块大小')
    parser.add_argument('-c', '--clipboard', action='store_true', help='从剪贴板获取')
    parser.add_argument('--accurate', action='store_true', help='使用高精度模型(更慢)')
    
    args = parser.parse_args()
    
    if args.clipboard or args.image is None:
        image_path = get_clipboard_image()
        if image_path is None:
            return
        if args.output is None:
            args.output = "mosaic_output.png"
    else:
        image_path = args.image
    
    mosaic = TextMosaic(use_fast_model=not args.accurate)
    mosaic.process_image(image_path, args.text, args.output, args.block_size)


if __name__ == '__main__':
    main()
