import os
import json
import cv2
import numpy as np
import fnmatch
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

root_dir = ".." ## 更改為你的路徑

def gen_mask_img(json_filename):
    """
    將 LabelMe JSON 轉換為 Segmentation Mask 影像
    """
    try:
        with open(json_filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return f"❌ [ERROR] 無法解析 JSON: {json_filename}"

    ### 獲取標註區域 (shapes)
    shapes = data.get("shapes", [])
    if not shapes:
        return f"⚠️ [WARNING] {json_filename} 沒有標註區域，跳過"

    target_dirname = os.path.dirname(json_filename)
    base_filename = Path(json_filename).stem

    #### 嘗試讀取對應影像
    original_img_filename = None
    for ext in [".jpg", ".png", ".jpeg"]:
        temp_path = os.path.join(target_dirname, base_filename + ext)
        if os.path.exists(temp_path):
            original_img_filename = temp_path
            break

    if original_img_filename is None:
        return f"⚠️ [WARNING] 找不到對應影像: {json_filename}，跳過"

    ### 讀取影像尺寸
    image = cv2.imread(original_img_filename)
    if image is None:
        return f"❌ [ERROR] 無法讀取影像: {original_img_filename}"
    height, width = image.shape[:2]

    ### 創建單通道 mask
    mask = np.zeros((height, width), dtype=np.uint8)

    ### 繪製所有標註區域
    for shape in shapes:
        points = shape.get("points", [])
        if not points:
            continue  # 跳過空的 shape
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=255)  # 255 代表前景

    ### 儲存 mask
    mask_img_filename = os.path.join(target_dirname, base_filename + "_mask.png")
    cv2.imwrite(mask_img_filename, mask)
    return f"✅ [INFO] 轉換成功: {mask_img_filename}"

def process_json_files_parallel():
    """
    使用多線程處理 JSON -> MASK 轉換
    """
    json_files = []
    for dirname, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if fnmatch.fnmatch(filename, "*.json"):
                json_files.append(os.path.join(dirname, filename).replace("\\", "/"))

    num_workers = min(cpu_count(), 24)  # 限制最多 24 線程，避免 CPU 過載 ， 視需求更改。
    print(f"🚀 啟動 {num_workers} 個線程，處理 {len(json_files)} 個 JSON 檔案...")

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(gen_mask_img, json_files), total=len(json_files)))

    # 輸出結果
    for res in results:
        if res:
            print(res)

if __name__ == '__main__':
    process_json_files_parallel()
