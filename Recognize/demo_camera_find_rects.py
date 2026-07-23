# demo_camera_find_rects.py - 摄像头实时找矩形示例
# 使用 sensor 获取图像 → Threshold → findContours → approxPolyDP 识别矩形

import time, os, gc, sys
from media.sensor import *
from media.display import *
from media.media import *
import cv2
from ulab import numpy as np

DETECT_WIDTH  = 320
DETECT_HEIGHT = 240

sensor = None

# 摄像头初始化
def init_camera():
    global sensor
    sensor = Sensor(width=1280, height=960, fps=90)                 #默认配置
    sensor.reset()
    sensor.set_framesize(width=640, height=480)  #调整实际输出尺寸
    sensor.set_pixformat(Sensor.RGB888)                             #像素格式
    sensor.run()
    time.sleep(0.5)

# 预处理 → 二值化 → 形态学去噪 → 找轮廓 → 几何筛选 → 可视化
def find_rectangles(img_np):
    """检测矩形轮廓并绘制"""
# 灰度化 + 模糊（预处理）
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 自适应阈值二值化
    binary = cv2.adaptiveThreshold(blurred, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

# 形态学闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 查找外部轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect_count = 0
    min_area = 300

# 轮廓筛选与矩形判定（核心）
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        if len(approx) == 4:
        # 绘制与标注
            cv2.drawContours(img_np, [approx], -1, (0, 255, 0), 3)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img_np, f"{w}x{h}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            rect_count += 1

    return rect_count

# 
def main():
    init_camera()
    Display.init(Display.ST7701, width=800, height=480, to_ide=True)

    fps_time = time.ticks_ms()
    frame_cnt = 0
    try:
        while True:
            img = sensor.snapshot()
            img_np = img.to_numpy_ref()
            n = find_rectangles(img_np)
            gc.collect()
            Display.show_image(img)

            frame_cnt += 1
            if time.ticks_diff(time.ticks_ms(), fps_time) >= 1000:
                fps = frame_cnt
                frame_cnt = 0
                fps_time = time.ticks_ms()
                print(f"FPS: {fps}, 检测到矩形: {n}")
    finally:
        if isinstance(sensor, Sensor):
            sensor.stop()
        Display.deinit()

if __name__ == "__main__":
    main()
