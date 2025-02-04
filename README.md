# Lableme-Annotation-Segment-Json-to-MASK
Lableme Annotation Segment Json to MASK
# Lableme-Annotation-Segment-Json-to-MASK
Lableme Annotation Segment Json to MASK

# LabelMe JSON to Mask Converter 🚀  

## 📝 介紹  
本專案提供 **Python 腳本**，可將 **LabelMe** 產生的 JSON 標註轉換為 **Segmentation Mask** (單通道 PNG)，方便後續其他語意分割模型訓練。

## 📂 目錄結構  
root_dir  
    ├── 00001.jpg
    ├── 00001.json
    ├── 00002.jpg
    ├── 00002.json
    ├── ..
    ├── ..
    ├── 00XXX.jpg
    ├── 00XXX.json


## ⚙️ **環境需求**  
本程式需 **Python 3.8+**，以及以下套件：  

```sh
pip install opencv-python numpy tqdm


