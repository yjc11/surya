from copy import deepcopy
from typing import List
from PIL import Image

from surya.detection import batch_text_detection
from surya.input.processing import slice_polys_from_image, slice_bboxes_from_image, convert_if_not_rgb
from surya.postprocessing.text import sort_text_lines
from surya.recognition import batch_recognition
from surya.schema import TextLine, OCRResult


def run_recognition(images: List[Image.Image], langs: List[List[str] | None], rec_model, rec_processor, bboxes: List[List[List[int]]] = None, polygons: List[List[List[List[int]]]] = None, batch_size=None) -> List[OCRResult]:
    # Polygons need to be in corner format - [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], bboxes in [x1, y1, x2, y2] format
    assert bboxes is not None or polygons is not None
    assert len(images) == len(langs), "You need to pass in one list of languages for each image"

    images = convert_if_not_rgb(images)

    slice_map = []
    all_slices = []
    all_langs = []
    for idx, (image, lang) in enumerate(zip(images, langs)):
        if polygons is not None:
            slices = slice_polys_from_image(image, polygons[idx])
        else:
            slices = slice_bboxes_from_image(image, bboxes[idx])
        slice_map.append(len(slices))
        all_slices.extend(slices)
        all_langs.extend([deepcopy(lang)] * len(slices))

    rec_predictions, _ = batch_recognition(all_slices, all_langs, rec_model, rec_processor, batch_size=batch_size)

    predictions_by_image = []
    slice_start = 0
    for idx, (image, lang) in enumerate(zip(images, langs)):
        slice_end = slice_start + slice_map[idx]
        image_lines = rec_predictions[slice_start:slice_end]
        slice_start = slice_end

        text_lines = []
        for i in range(len(image_lines)):
            if polygons is not None:
                poly = polygons[idx][i]
            else:
                bbox = bboxes[idx][i]
                poly = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]

            text_lines.append(TextLine(
                text=image_lines[i],
                polygon=poly
            ))

        pred = OCRResult(
            text_lines=text_lines,
            languages=lang,
            image_bbox=[0, 0, image.size[0], image.size[1]]
        )
        predictions_by_image.append(pred)

    return predictions_by_image


def run_ocr(images: List[Image.Image], langs: List[List[str] | None], det_model, det_processor, rec_model, rec_processor, batch_size=None, highres_images: List[Image.Image] | None = None) -> List[OCRResult]:
    # 确保所有图像都是RGB格式
    images = convert_if_not_rgb(images)
    # 如果提供了高分辨率图像,也将其转换为RGB格式;否则,用None填充
    highres_images = convert_if_not_rgb(highres_images) if highres_images is not None else [None] * len(images)
    # 对所有图像进行文本检测
    det_predictions = batch_text_detection(images, det_model, det_processor)

    all_slices = []  # 存储所有图像的文本区域切片
    slice_map = []   # 记录每张图像的文本区域数量
    all_langs = []   # 存储所有文本区域对应的语言

    # 遍历每张图像及其对应的检测结果、高分辨率图像(如果有)和语言
    for idx, (det_pred, image, highres_image, lang) in enumerate(zip(det_predictions, images, highres_images, langs)):
        # 获取检测到的多边形区域
        polygons = [p.polygon for p in det_pred.bboxes]
        if highres_image:
            # 如果有高分辨率图像,计算缩放比例
            width_scaler = highres_image.size[0] / image.size[0]
            height_scaler = highres_image.size[1] / image.size[1]
            # 将多边形坐标缩放到高分辨率图像的尺寸
            scaled_polygons = [[[int(p[0] * width_scaler), int(p[1] * height_scaler)] for p in polygon] for polygon in polygons]
            # 从高分辨率图像中切割文本区域
            slices = slice_polys_from_image(highres_image, scaled_polygons)
        else:
            # 如果没有高分辨率图像,直接从原图中切割文本区域
            slices = slice_polys_from_image(image, polygons)
        
        # 记录该图像的文本区域数量
        slice_map.append(len(slices))
        # 为每个文本区域添加对应的语言
        all_langs.extend([lang] * len(slices))
        # 将该图像的所有文本区域切片添加到总列表中
        all_slices.extend(slices)

    # 对所有文本区域进行批量识别
    rec_predictions, confidence_scores = batch_recognition(all_slices, all_langs, rec_model, rec_processor, batch_size=batch_size)

    predictions_by_image = []  # 存储每张图像的OCR结果
    slice_start = 0  # 记录当前图像的文本区域在总列表中的起始位置
    # 遍历每张图像及其对应的检测结果和语言
    for idx, (image, det_pred, lang) in enumerate(zip(images, det_predictions, langs)):
        # 计算当前图像的文本区域在总列表中的结束位置
        slice_end = slice_start + slice_map[idx]
        # 获取当前图像的文本识别结果和置信度
        image_lines = rec_predictions[slice_start:slice_end]
        line_confidences = confidence_scores[slice_start:slice_end]
        # 更新下一张图像的起始位置
        slice_start = slice_end

        # 确保识别结果数量与检测到的文本框数量一致
        assert len(image_lines) == len(det_pred.bboxes)

        lines = []
        # 将识别结果、置信度和边界框信息组合
        for text_line, confidence, bbox in zip(image_lines, line_confidences, det_pred.bboxes):
            lines.append(TextLine(
                text=text_line,
                polygon=bbox.polygon,
                bbox=bbox.bbox,
                confidence=confidence
            ))

        # 对文本行进行排序(可能是基于位置)
        lines = sort_text_lines(lines)

        # 创建并添加当前图像的OCR结果
        predictions_by_image.append(OCRResult(
            text_lines=lines,
            languages=lang,
            image_bbox=det_pred.image_bbox
        ))

    # 返回所有图像的OCR结果
    return predictions_by_image
