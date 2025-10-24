
import cv2
import os
import numpy as np
import cv2
import os
import numpy as np

def resize_keep_aspect(image, height=None):
    h0, w0 = image.shape[:2]
    if height is not None:
        scale = height / float(h0)
        return cv2.resize(image, (int(w0*scale), int(height)))
    return image.copy()

def detect_ui_regions_auto(image_path, output_dir="output_regions", debug=True, target_h=600):
    os.makedirs(output_dir, exist_ok=True)

    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"无法加载图片: {image_path}")

    # 缩放图用于检测
    resized = resize_keep_aspect(orig, height=target_h)
    rh, rw = resized.shape[:2]
    oh, ow = orig.shape[:2]

    # =====  边界增强 =====
    border = int(max(rh, rw)*0.02)  # 边界占比 2%
    padded = cv2.copyMakeBorder(resized, border, border, border, border, cv2.BORDER_REPLICATE)

    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray0", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    gray = cv2.bilateralFilter(gray, 2, 75, 20)

    # ===== 自适应阈值 =====
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 8)

    # =====  动态核大小 =====
    scale_factor = max(rh, rw)/10000  # 按缩放图尺寸计算
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3,int(3*scale_factor)), max(3,int(3*scale_factor))))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15,int(25*scale_factor)), max(5,int(7*scale_factor))))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10,int(15*scale_factor)), max(5,int(5*scale_factor))))

    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_open)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close)
    th = cv2.dilate(th, kernel_dilate, iterations=1)
    cv2.imshow("gaussian", th)
    cv2.imshow("gray1", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # =====  连通域检测 =====
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    candidates = []
    for i in range(1, num_labels):
        x, y, w_box, h_box, area = stats[i]
        if area < max(500, 0.005*rw*rh):  # 面积过滤
            continue
        # 去掉 padding 偏移
        x = max(0, x-border)
        y = max(0, y-border)
        candidates.append((x, y, w_box, h_box, area))

    candidates = sorted(candidates, key=lambda r: r[1])
    vis_resized = resized.copy()
    vis_orig = orig.copy()

    scale_x = ow / float(rw)
    scale_y = oh / float(rh)
    idx = 0
    results = {}

    for (x, y, w_box, h_box, area) in candidates:
        cv2.rectangle(vis_resized, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
        x_o = int(round(x*scale_x))
        y_o = int(round(y*scale_y))
        w_o = int(round(w_box*scale_x))
        h_o = int(round(h_box*scale_y))

        # 边界限制
        x_o = max(0, min(x_o, ow-1))
        y_o = max(0, min(y_o, oh-1))
        if x_o + w_o > ow:
            w_o = ow - x_o
        if y_o + h_o > oh:
            h_o = oh - y_o

        cv2.rectangle(vis_orig, (x_o, y_o), (x_o+w_o, y_o+h_o), (0, 0, 255), 2)
        roi = orig[y_o:y_o+h_o, x_o:x_o+w_o]

        save_path = os.path.join(output_dir, f"region_{idx}.png")
        cv2.imwrite(save_path, roi)
        results[f"region_{idx}"] = (x_o, y_o, w_o, h_o)
        idx += 1

    if debug:
        cv2.imwrite(os.path.join(output_dir, "binary_with_padding.png"), th)
        cv2.imwrite(os.path.join(output_dir, "resized_with_boxes.png"), vis_resized)
        cv2.imwrite(os.path.join(output_dir, "orig_with_boxes.png"), vis_orig)

    print(f"检测完成: {len(results)} 个区域")
    return results

# quick test
if __name__ == "__main__":
    image_path = r"E:\workspace\doc\img\test.png"
    output_dir = "output_regions"
    detect_ui_regions_auto(image_path, output_dir=output_dir, debug=True)

