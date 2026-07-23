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
    sensor.set_framesize(width=640, height=480)                     #调整实际输出尺寸
    sensor.set_pixformat(Sensor.RGB888)                             #像素格式
    sensor.run()
    time.sleep(0.5)
    
def box_points(rect):
    """根据 cv2.minAreaRect 的返回值,手动计算四个角点(纯 Python 实现,兼容 K230)"""
    center, size, angle = rect
    cx, cy = center
    w, h = size

    # 角度转弧度
    import math
    theta = angle * math.pi / 180.0
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # 未旋转时的四个角点偏移量
    offsets = [
        [-w/2, -h/2],
        [ w/2, -h/2],
        [ w/2,  h/2],
        [-w/2,  h/2]
    ]

    # 手动旋转并加上中心点坐标
    rotated = []
    for x_off, y_off in offsets:
        x_rot = x_off * cos_t - y_off * sin_t
        y_rot = x_off * sin_t + y_off * cos_t
        rotated.append([int(round(cx + x_rot)), int(round(cy + y_rot))])

    # 返回列表,cv2.drawContours 可以直接用
    return rotated


# 预处理 → 二值化 → 形态学去噪 → 找轮廓 → 几何筛选 → 可视化
def find_rectangles(img_np):
    """检测矩形轮廓并绘制"""
# 灰度化 + 模糊(预处理)
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)                 # 灰度化适用于只看形状、纹理
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)                     # 模糊化有利于降低噪声,(3,3)->增加细节

# 自适应阈值二值化
    """
    def adaptiveThreshold(src: Any, maxValue: Any, adaptiveMethod: int, thresholdType: int, blockSize: int, C: Any, dst: Any = None) -> Any:
    需要调整的:
    blockSize:  必须是奇数,值小能抓出细腻的细节,放大噪声;值大抵抗噪声,目标边缘糊掉
    → 如果画面里目标很大,可以调大(比如15、21),让阈值更稳定.
    → 如果目标很小很精细,调小(比如7、9),避免被当做背景过滤掉.

    2(C值): "比局部平均亮度低多少才算目标".
    → 稍微暗一些的目标都提取,减小这个值(比如0或1).
    → 只提取特别暗的目标,就增大(比如5、10).
    """
    binary = cv2.adaptiveThreshold(blurred, 255,                    
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

# 形态学闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 对于二值化后的修补功能,可替换不同形状的"刷子",刷子大小改(5, 5)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 查找外部轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #改closed和上一行代码相关

    mushroom_count  = 0
    min_area = 300  #根据下面测试的结果填像素点

# 轮廓筛选与矩形判定(核心)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area < min_area:
            continue

        # ------- 形状判定:改成圆形度 -------
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circularity = 4 * 3.14159 * area / (peri * peri)
        if circularity < 0.4:   # 蘑菇通常不完全是圆形,阈值设低一点
            continue
        # -----------------------------------

        # ------- 绘制:用最小外接斜矩形 -------
        if len(cnt) >= 4:
            rect = cv2.minAreaRect(cnt)
            box = box_points(rect)
            cv2.drawContours(img_np, [box], 0, (0, 255, 0), 3)   # 绿色斜矩形框
        
        # 可选:同时保留正矩形和文字标注
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img_np, "Mushroom", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        mushroom_count += 1

    return mushroom_count

 
def main():
    init_camera()
    # Display.init(Display.ST7701, width=800, height=480, to_ide=True, rotation=180)
    Display.init(Display.ST7701, width=800, height=480, to_ide=True)

    fps_time = time.ticks_ms()
    frame_cnt = 0
    try:
        # 取图 → 转换 → 处理 → 清理内存 → 显示
        while True:
            img = sensor.snapshot()
            img_np = img.to_numpy_ref()
            n = find_rectangles(img_np)
            gc.collect()
            Display.show_image(img)

        # 帧率统计(每秒输出一次)
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

# 保证它作为主程序时候可以运行,然后作为被调用的函数时候不运行,但是里面的函数可以被调用者使用
if __name__ == "__main__":
    main()
