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
from PyQt5.QtWidgets import (  # 新增：用于添加勾选框和分组
    QApplication,
    QCheckBox,
    QFrame,
    QGroupBox,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from surya.apis.baidu_translate import baidu_translate
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


class OCRProcessor:
    def __init__(self):
        self.det_model, self.det_processor = load_det_cached()
        self.rec_model, self.rec_processor = load_rec_cached()

    def perform_ocr(self, image: Image.Image, langs: List[str]) -> OCRResult:
        # Adjust image size
        image = adjust_image_size(image)
        # Perform OCR
        _, ocr_result = ocr(
            image,
            image,
            langs,
            self.det_model,
            self.det_processor,
            self.rec_model,
            self.rec_processor,
        )
        return ocr_result


class Translator:
    def __init__(self):
        pass

    def translate_text(self, text: str, from_lang: str = "en", to_lang: str = "zh") -> str:
        try:
            text = text.replace("\n", " ")
            return baidu_translate(text, from_lang=from_lang, to_lang=to_lang)
        except Exception as e:
            print(f"Translation error: {e}")
            return "Error: Unable to translate text"


class DesktopOCRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.ocr_processor = OCRProcessor()
        self.translator = Translator()
        self.image = None

    def initUI(self):
        self.setWindowTitle('OCR & Translate App')
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon('path_to_your_icon.png'))

        # 创建中央部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # OCR 部分
        ocr_frame = self.create_ocr_frame()
        main_layout.addWidget(ocr_frame)

        # 设置全局样式
        self.set_global_style()

    def create_ocr_frame(self):
        # 创建OCR框架
        ocr_frame = QFrame()
        ocr_frame.setFrameShape(QFrame.StyledPanel)
        ocr_layout = QVBoxLayout(ocr_frame)

        # 添加OCR标题
        ocr_title = QLabel("OCR")
        ocr_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        ocr_layout.addWidget(ocr_title)

        # 添加按钮布局
        button_layout = self.create_button_layout()
        ocr_layout.addLayout(button_layout)

        # 添加翻译复选框
        self.translate_checkbox = self.create_translate_checkbox()
        ocr_layout.addWidget(self.translate_checkbox)

        # 添加图片显示区域
        image_scroll_area = self.create_image_scroll_area()
        ocr_layout.addWidget(image_scroll_area)

        # 添加结果显示区域
        results_group = self.create_results_group()
        ocr_layout.addWidget(results_group)

        return ocr_frame

    def create_button_layout(self):
        # 创建按钮布局
        button_layout = QHBoxLayout()

        # 添加粘贴图片按钮
        self.paste_button = QPushButton('Paste Image')
        self.paste_button.clicked.connect(self.paste_image)
        button_layout.addWidget(self.paste_button)

        # 添加执行OCR按钮
        self.ocr_button = QPushButton('Perform OCR')
        self.ocr_button.clicked.connect(self.perform_ocr)
        button_layout.addWidget(self.ocr_button)

        return button_layout

    def create_translate_checkbox(self):
        # 创建翻译复选框
        translate_checkbox = QCheckBox('OCR后立刻翻译')
        translate_checkbox.setStyleSheet(
            """
            QCheckBox {
                color: #333333;
                font-size: 14px;
                padding: 5px;
            }
            QCheckBox::indicator {
                color: #2196F3;
                width: 18px;
                height: 18px;
            }
        """
        )
        return translate_checkbox

    def create_image_scroll_area(self):
        # 创建图片显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 10px solid #ddd;")

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)

        return scroll_area

    def create_results_group(self):
        # 创建结果显示区域
        results_group = QGroupBox("Results")
        results_layout = QHBoxLayout(results_group)

        # OCR结果显示
        ocr_result_layout = self.create_ocr_result_layout()
        results_layout.addLayout(ocr_result_layout)

        # 翻译结果显示
        translate_result_layout = self.create_translate_result_layout()
        results_layout.addLayout(translate_result_layout)

        return results_group

    def create_ocr_result_layout(self):
        # 创建OCR结果布局
        ocr_result_layout = QVBoxLayout()
        ocr_result_label = QLabel("OCR Result:")
        ocr_result_layout.addWidget(ocr_result_label)

        self.result_text = QTextEdit()
        self.result_text.setPlaceholderText("OCR result will be displayed here...")
        ocr_result_layout.addWidget(self.result_text)

        return ocr_result_layout

    def create_translate_result_layout(self):
        # 创建翻译结果布局
        translate_result_layout = QVBoxLayout()
        translate_result_label = QLabel("Translated Text:")
        translate_result_layout.addWidget(translate_result_label)

        self.translated_text = QTextEdit()
        self.translated_text.setPlaceholderText("Translated text will appear here...")
        self.translated_text.setReadOnly(True)
        translate_result_layout.addWidget(self.translated_text)

        # 添加单独的翻译按钮
        self.translate_button = QPushButton('翻译')
        self.translate_button.clicked.connect(self.translate_only)
        translate_result_layout.addWidget(self.translate_button)

        return translate_result_layout

    def set_global_style(self):
        # 设置全局样式
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f0f0f0;
            }
            QFrame {
                background-color: #ffffff;
                border-radius: 5px;
                border: 1px solid #e0e0e0;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QTextEdit {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #bdbdbd;
                border-radius: 3px;
                padding: 5px;
                font-size: 12px;
            }
            QLabel {
                color: #333333;
                font-size: 12px;
            }
            QCheckBox {
                color: #333333;
                font-size: 14px;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
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
        pil_image = self.convert_qimage_to_pil(self.image)

        # Perform OCR
        ocr_result = self.ocr_processor.perform_ocr(pil_image, ["English", "Chinese"])

        # Display OCR results
        text_lines = [line.text for line in ocr_result.text_lines]
        ocr_text = "\n".join(text_lines)
        self.result_text.setText(ocr_text)

        # Translate if checkbox is checked
        if self.translate_checkbox.isChecked():
            self.translate_text()

    def translate_text(self):
        ocr_text = self.result_text.toPlainText()
        if not ocr_text:
            self.translated_text.setText("No text to translate.")
            return

        translated = self.translator.translate_text(ocr_text)
        self.translated_text.setText(translated)

    def translate_only(self):
        self.translate_text()

    def convert_qimage_to_pil(self, qimage: QImage) -> Image.Image:
        buffer = QPixmap.fromImage(qimage).toImage()
        buffer = buffer.convertToFormat(QImage.Format_RGB888)
        width, height = buffer.width(), buffer.height()
        ptr = buffer.constBits()
        ptr.setsize(height * width * 3)
        return Image.frombytes("RGB", (width, height), ptr, "raw", "RGB")


def main():
    app = QApplication(sys.argv)
    ex = DesktopOCRApp()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
