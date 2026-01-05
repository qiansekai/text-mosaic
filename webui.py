# -*- coding: utf-8 -*-
"""
文字打码 WebUI - Gradio 版
支持粘贴图片、拖拽上传、实时预览
"""

import gradio as gr
import cv2
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'

os.environ['no_proxy'] = 'localhost,127.0.0.1'

from paddleocr import PaddleOCR

# 全局 OCR 实例
ocr = None

def load_ocr():
    global ocr
    if ocr is None:
        print("正在加载 OCR 模型...")
        ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=True,
        )
        print("模型加载完成!")
    return ocr

def apply_mosaic(image, box, block_size=10):
    x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    h, w = image.shape[:2]
    x_min, x_max = max(0, x_min), min(w, x_max)
    y_min, y_max = max(0, y_min), min(h, y_max)
    
    if x_max <= x_min or y_max <= y_min:
        return
    
    roi = image[y_min:y_max, x_min:x_max]
    roi_h, roi_w = roi.shape[:2]
    bs = max(1, min(block_size, roi_h, roi_w))
    
    small = cv2.resize(roi, (max(1, roi_w // bs), max(1, roi_h // bs)))
    image[y_min:y_max, x_min:x_max] = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

def process_image(image, target_text, block_size):
    if image is None:
        return None, "请上传或粘贴图片"
    
    if not target_text.strip():
        return image, "请输入要打码的文字"
    
    ocr = load_ocr()
    
    # Gradio 传入的是 RGB，转 BGR 处理
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 保存临时文件给 OCR
    temp_path = "temp_input.jpg"
    cv2.imwrite(temp_path, image_bgr)
    
    results = ocr.predict(temp_path, return_word_box=True)
    
    target_texts = [t.strip() for t in target_text.split(',') if t.strip()]
    target_set = set(target_texts)
    mosaic_count = 0
    
    for res in results:
        rec_texts = res.get('rec_texts', [])
        text_words = res.get('text_word', [])
        text_word_boxes = res.get('text_word_boxes', [])
        
        if not text_words or not text_word_boxes:
            continue
        
        for line_text, words, word_boxes in zip(rec_texts, text_words, text_word_boxes):
            if not words or word_boxes is None:
                continue
            if not any(t in line_text for t in target_set):
                continue
            
            word_boxes_list = word_boxes.tolist() if hasattr(word_boxes, 'tolist') else word_boxes
            combined = ''.join(words)
            
            for target in target_set:
                if target not in combined:
                    continue
                
                start = 0
                while True:
                    pos = combined.find(target, start)
                    if pos == -1:
                        break
                    
                    char_count = 0
                    target_end = pos + len(target)
                    
                    for word_idx, word in enumerate(words):
                        word_end = char_count + len(word)
                        if word_end > pos and char_count < target_end and word_idx < len(word_boxes_list):
                            apply_mosaic(image_bgr, word_boxes_list[word_idx], block_size)
                            mosaic_count += 1
                        char_count = word_end
                    
                    start = pos + 1
    
    # 清理临时文件
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    # 转回 RGB 显示
    result_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return result_rgb, f"完成！共打码 {mosaic_count} 处"

# 预加载模型
print("启动 WebUI...")
load_ocr()

# 创建界面
with gr.Blocks(title="文字打码工具") as demo:
    gr.Markdown("# 📝 文字打码工具")
    gr.Markdown("上传图片或 Ctrl+V 粘贴，输入要打码的文字，支持多个文字用逗号分隔")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="输入图片", type="numpy")
            target_text = gr.Textbox(label="要打码的文字", placeholder="例如: 牛角")
            block_size = gr.Slider(5, 30, value=10, step=1, label="马赛克块大小")
            submit_btn = gr.Button("开始打码", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="处理结果")
            status = gr.Textbox(label="状态")
    
    submit_btn.click(
        fn=process_image,
        inputs=[input_image, target_text, block_size],
        outputs=[output_image, status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
