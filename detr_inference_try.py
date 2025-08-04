import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import time

# 載入圖片（換成你的圖片路徑）
image = Image.open("local_deformed_fin.jpg").convert("RGB")

# 載入模型與前處理
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

def benchmark(device, repeat=10):
    model.to(device)
    inputs = processor(images=image, return_tensors="pt").to(device)

    # 預熱一次
    with torch.no_grad():
        _ = model(**inputs)

    # 正式測速多次平均
    total_time = 0
    for _ in range(repeat):
        start = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        end = time.time()
        total_time += (end - start)

    avg_time = total_time / repeat
    fps = 1 / avg_time
    print(f"📍 Device: {device}")
    print(f"🕒 Average Inference Time: {avg_time:.4f} seconds")
    print(f"🎯 FPS: {fps:.2f} frames/sec\n")

# 判斷 GPU 是否可用
if torch.cuda.is_available():
    benchmark(torch.device("cuda"))
else:
    print("⚠️ 沒有偵測到 GPU，跳過 GPU 測速")

benchmark(torch.device("cpu"))
