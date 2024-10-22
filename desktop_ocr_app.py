import io
import sys
from typing import List, Union

import pypdfium2
import requests  # 新增：用于发送HTTP请求

# import streamlit as st
from PIL import Image, ImageGrab
from pypdfium2 import PdfiumError
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QHBoxLayout  # 新增：用于水平布局
from PyQt5.QtWidgets import QLabel  # 新增：用于添加标签
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from surya.detection import batch_text_detection
from surya.input.langs import replace_lang_with_code
from surya.input.pdflines import get_page_text_lines, get_table_blocks
from surya.languages import CODE_TO_LANGUAGE
from surya.layout import batch_layout_detection
from surya.model.detection.model import load_model, load_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.ocr import run_ocr
from surya.ordering import batch_ordering
from surya.postprocessing.heatmap import draw_bboxes_on_image, draw_polys_on_image
from surya.postprocessing.text import draw_text_on_image
from surya.postprocessing.util import rescale_bbox, rescale_bboxes
from surya.schema import (
    LayoutResult,
    OCRResult,
    OrderResult,
    TableResult,
    TextDetectionResult,
)
from surya.settings import settings


def load_det_cached():
    checkpoint = settings.DETECTOR_MODEL_CHECKPOINT
    return load_model(checkpoint=checkpoint), load_processor(checkpoint=checkpoint)


def load_rec_cached():
    return load_rec_model(), load_rec_processor()


# Function for OCR
def ocr(
    img,
    highres_img,
    langs: List[str],
    det_model,
    det_processor,
    rec_model,
    rec_processor,
) -> Union[Image.Image, OCRResult]:
    replace_lang_with_code(langs)
    img_pred = run_ocr(
        [img],
        [langs],
        det_model,
        det_processor,
        rec_model,
        rec_processor,
        highres_images=[highres_img],
    )[0]

    bboxes = [l.bbox for l in img_pred.text_lines]
    text = [l.text for l in img_pred.text_lines]
    rec_img = draw_text_on_image(bboxes, text, img.size, langs, has_math="_math" in langs)
    print(img_pred)
    return rec_img, img_pred


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


def get_page_image(pdf_file, page_num, dpi=settings.IMAGE_DPI):
    doc = open_pdf(pdf_file)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    png_image = png.convert("RGB")
    return png_image


def page_count(pdf_file):
    doc = open_pdf(pdf_file)
    return len(doc)


def adjust_image_size(image: Image.Image, target_width: int = 2048) -> Image.Image:
    """
    Adjust the image size to have a width of 2048 pixels if it's smaller,
    or scale it down if it's larger while maintaining the aspect ratio.
    """
    width, height = image.size
    if width == target_width:
        return image

    aspect_ratio = height / width
    new_height = int(target_width * aspect_ratio)

    return image.resize((target_width, new_height), Image.LANCZOS)


def polish_text(text: str) -> str:
    """
    发送文本到模型务进行润色
    """
    # 这里替换为实际的模型服务URL
    url = "https://your-model-service-url.com/polish"

    try:
        response = requests.post(url, json={"text": text})
        response.raise_for_status()
        return response.json()["polished_text"]
    except requests.RequestException as e:
        print(f"Error occurred while polishing text: {e}")
        return "Error: Unable to polish text"


class DesktopOCRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Load OCR models
        self.det_model, self.det_processor = load_det_cached()
        self.rec_model, self.rec_processor = load_rec_cached()

    def initUI(self):
        self.setWindowTitle('OCR & Text Polish App')
        self.setGeometry(100, 100, 1200, 800)  # 增加窗口大小
        self.setWindowIcon(QIcon('path_to_your_icon.png'))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()

        # OCR 部分
        ocr_frame = QFrame()
        ocr_frame.setFrameShape(QFrame.StyledPanel)
        ocr_layout = QVBoxLayout(ocr_frame)

        ocr_title = QLabel("OCR")
        ocr_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        ocr_layout.addWidget(ocr_title)

        self.paste_button = QPushButton('Paste Image')
        self.paste_button.clicked.connect(self.paste_image)
        ocr_layout.addWidget(self.paste_button)

        self.ocr_button = QPushButton('Perform OCR')
        self.ocr_button.clicked.connect(self.perform_ocr)
        ocr_layout.addWidget(self.ocr_button)

        # 添加图片显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd;")

        # 使用QScrollArea来使图片可滚动
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        ocr_layout.addWidget(scroll_area)

        self.result_text = QTextEdit()
        self.result_text.setPlaceholderText("OCR result will be displayed here...")
        ocr_layout.addWidget(self.result_text)

        # 文本润色部分
        polish_frame = QFrame()
        polish_frame.setFrameShape(QFrame.StyledPanel)
        polish_layout = QVBoxLayout(polish_frame)

        polish_title = QLabel("Text Polish")
        polish_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        polish_layout.addWidget(polish_title)

        self.polish_input = QTextEdit()
        self.polish_input.setPlaceholderText("Enter text to polish...")
        polish_layout.addWidget(self.polish_input)

        self.polish_button = QPushButton('Polish Text')
        self.polish_button.clicked.connect(self.polish_text)
        polish_layout.addWidget(self.polish_button)

        self.polished_result = QTextEdit()
        self.polished_result.setReadOnly(True)
        self.polished_result.setPlaceholderText("Polished text will appear here...")
        polish_layout.addWidget(self.polished_result)

        # 使用QSplitter来允许用户调整两个部分的大小
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(ocr_frame)
        splitter.addWidget(polish_frame)
        splitter.setSizes([600, 600])  # 调整初始大小

        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)

        self.image = None

        # 设置全局样式
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f0f0f0;
            }
            QFrame {
                background-color: #ffffff;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QTextEdit {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #bdbdbd;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
            }
            QLabel {
                color: #333333;
            }
        """
        )

    def paste_image(self):
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()

        if mime_data.hasImage():
            image = QImage(mime_data.imageData())
            if not image.isNull():
                self.image = image
                self.display_image(image)
                self.result_text.setText(
                    "Image pasted from clipboard. Click 'Perform OCR' to process."
                )
            else:
                self.result_text.setText("Failed to paste image from clipboard.")
        else:
            self.result_text.setText("No image found in clipboard.")

    def display_image(self, image):
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def perform_ocr(self):
        if self.image is None:
            self.result_text.setText("Please paste an image first.")
            return

        # Convert QImage to PIL Image
        buffer = QPixmap.fromImage(self.image).toImage()
        buffer = buffer.convertToFormat(QImage.Format_RGB888)
        width, height = buffer.width(), buffer.height()
        ptr = buffer.constBits()
        ptr.setsize(height * width * 3)
        pil_image = Image.frombytes("RGB", (width, height), ptr, "raw", "RGB")

        # Adjust image size
        pil_image = adjust_image_size(pil_image)

        # Perform OCR
        _, ocr_result = ocr(
            pil_image,
            pil_image,
            ["English", "Chinese"],
            self.det_model,
            self.det_processor,
            self.rec_model,
            self.rec_processor,
        )

        # Display results
        text_lines = [line.text for line in ocr_result.text_lines]
        self.result_text.setText("\n".join(text_lines))

    def polish_text(self):
        input_text = self.polish_input.toPlainText()
        if not input_text:
            self.polished_result.setText("Please enter some text to polish.")
            return

        polished_text = polish_text(input_text)
        self.polished_result.setText(polished_text)


def main():
    app = QApplication(sys.argv)
    ex = DesktopOCRApp()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
