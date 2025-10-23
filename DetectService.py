import cv2

import numpy as np
def detect_by_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 形态学操作连接边缘
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 填充轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, -1)  # 填充轮廓

    return mask


def detect_ui_components(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. 使用更温和的边缘检测参数
    edges = cv2.Canny(gray, 30, 100)  # 降低阈值

    # 2. 使用形态学操作连接相近的区域
    # 水平方向连接（连接同一行的文字）
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_horizontal)

    # 垂直方向连接（连接相关区域）
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel_vertical)

    # 3. 进一步填充区域
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel_fill)

    # 4. 查找轮廓 - 只获取最外层轮廓
    contours, _ = cv2.findContours(connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def detect_text_lightweight(image):
    """
    轻量级文字检测（组合多种方法）
    """
    # 方法1：基于轮廓的检测
    contour_result = has_text_contour_based(image)

    # 方法2：基于边缘密度的检测
    edge_result = has_text_edge_density(image)

    # 如果两种方法都认为有文字，则返回True
    return contour_result or edge_result


def has_text_edge_density(image):
    """
    基于边缘密度的文字检测
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 计算边缘密度
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    # 文字区域通常有较高的边缘密度
    # 阈值可以根据实际情况调整
    return edge_density > 0.01  # 1%的边缘密度


def has_text_contour_based(image):
    """
    基于轮廓的文字区域检测
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 图像预处理
    # 1. 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. 自适应阈值
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 3. 形态学操作连接文字区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 4. 查找轮廓
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. 分析轮廓特征（文字区域通常有特定特征）
    text_like_contours = 0

    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)

        # 计算边界矩形
        x, y, w, h = cv2.boundingRect(contour)

        # 计算宽高比（文字通常有特定的宽高比）
        aspect_ratio = w / h if h > 0 else 0

        # 计算轮廓面积与边界矩形面积的比例
        rect_area = w * h
        area_ratio = area / rect_area if rect_area > 0 else 0

        # 文字区域的特征：
        # - 面积适中（不是太大也不是太小）
        # - 宽高比通常在0.1到10之间
        # - 轮廓相对紧凑
        if (area > 20 and area < 5000 and  # 面积范围
                0.1 < aspect_ratio < 10 and  # 宽高比范围
                area_ratio > 0.2):  # 紧凑度
            text_like_contours += 1

    # 如果有多个文字特征的轮廓，则认为包含文字
    return text_like_contours >= 3


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
   #计算透视变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    #把原始图像中的四边形区域“拉伸变换”成矩形，得到新的“俯视视角”图像。
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized