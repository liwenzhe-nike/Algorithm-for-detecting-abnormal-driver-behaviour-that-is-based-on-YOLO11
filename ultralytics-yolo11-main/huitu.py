import cv2
import numpy as np
import os

def swap_br_corner(img1_path, img2_path, out_dir='.'):
    """交换两张图片右下角 1/4 区域"""
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError('图片路径错误或图片无法读取')

    # 统一尺寸（按小的来）
    h, w = min(img1.shape[:2], img2.shape[:2])
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))

    # 右下角 1/4 区域
    # ========= 右上角 1/4 区域 =========
    y0, x0 = 0, w // 2  # 右上角
    ur1 = img1[y0:h // 2, x0:].copy()
    ur2 = img2[y0:h // 2, x0:].copy()

    # 交换
    img1[y0:h // 2, x0:] = ur2
    img2[y0:h // 2, x0:] = ur1

    # 交换


    # 保存
    out1 = os.path.join(out_dir, 'chaojia.png')
    out2 = os.path.join(out_dir, 'chaojib.png')
    cv2.imwrite(out1, img1)
    cv2.imwrite(out2, img2)
    print(f'已保存：{out1}  &  {out2}')

if __name__ == '__main__':
    swap_br_corner('A_swapped.png', 'B_swapped.png')          # ← 改成你的文件名