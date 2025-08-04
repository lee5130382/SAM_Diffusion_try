import cv2
import os
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamPredictor
import time
from datetime import datetime
import yaml
print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA 版本:", torch.version.cuda)
    print("cuDNN 版本:", torch.backends.cudnn.version())
    print("GPU 數量:", torch.cuda.device_count())
    print("GPU 名稱:", torch.cuda.get_device_name(0))




### 載入配置文件
def load_config(config_path="config.yaml"):
    """載入 YAML 配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        print(f"✅ 成功載入配置文件: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ 找不到配置文件: {config_path}")
        print("正在創建預設配置文件...")
        create_default_config(config_path)
        return load_config(config_path)
    except yaml.YAMLError as e:
        print(f"❌ 配置文件格式錯誤: {e}")
        raise

def create_default_config(config_path="config.yaml"):
    """創建預設配置文件"""
    default_config = {
        'paths': {
            'video_path': 'D:/project/diffusion/vedio/test.mp4',
            'frames_dir': 'frames',
            'masks_dir': 'masks',
            'recolored_dir': 'recolored_frames',
            'output_video': 'output_video.mp4'
        },
        'sam': {
            'model_type': 'vit_b',
            'checkpoint_path': 'sam_model/sam_vit_b_01ec64.pth'
        },
        'processing': {
            'target_color': [0, 0, 0],
            'video_codec': 'mp4v',
            'use_cuda': True
        },
        'display': {
            'progress_interval': 30,
            'show_preview': True,
            'window_title': '請點選目標物件 (點選後按任意鍵)'
        },
        'optimization': {
            'batch_size': 1,
            'memory_limit_gb': 8
        },
        'output': {
            'preserve_fps': True,
            'quality': 'high'
        },
        'debug': {
            'verbose': True,
            'save_intermediate': True,
            'log_timing': True
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as file:
        yaml.dump(default_config, file, default_flow_style=False, allow_unicode=True, indent=2)
    print(f"✅ 已創建預設配置文件: {config_path}")

### 性能監測裝飾器
def time_it(func_name, config):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not config['debug']['log_timing']:
                return func(*args, **kwargs)
                
            start_time = time.time()
            if config['debug']['verbose']:
                print(f"⏱️ 開始執行 {func_name}...")
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            elapsed = end_time - start_time
            if config['debug']['verbose']:
                print(f"✅ {func_name} 完成 - 耗時: {elapsed:.2f} 秒")
            return result
        return wrapper
    return decorator

# 全域配置
CONFIG = load_config()

# 設備檢測
use_cuda = CONFIG['processing']['use_cuda'] and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if CONFIG['debug']['verbose']:
    print(f"🖥️ 使用設備: {device}")
    if use_cuda:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

# 設定參數類別 (為了向下相容)
class Config:
    def __init__(self, config_dict):
        # 路徑設定
        self.VIDEO_PATH = config_dict['paths']['video_path']
        self.FRAMES_DIR = config_dict['paths']['frames_dir']
        self.MASKS_DIR = config_dict['paths']['masks_dir']
        self.RECOLORED_DIR = config_dict['paths']['recolored_dir']
        self.OUTPUT_VIDEO = config_dict['paths']['output_video']
        
        # SAM 設定
        self.MODEL_TYPE = config_dict['sam']['model_type']
        self.CHECKPOINT_PATH = config_dict['sam']['checkpoint_path']
        
        # 處理設定
        self.TARGET_COLOR = config_dict['processing']['target_color']
        self.VIDEO_CODEC = config_dict['processing']['video_codec']
        
        # 顯示設定
        self.PROGRESS_INTERVAL = config_dict['display']['progress_interval']
        self.WINDOW_TITLE = config_dict['display']['window_title']
        self.SHOW_PREVIEW = config_dict['display']['show_preview']

# 初始化配置實例
config = Config(CONFIG)

# 創建必要的目錄
def setup_directories():
    dirs = [config.FRAMES_DIR, config.MASKS_DIR, config.RECOLORED_DIR]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    if CONFIG['debug']['verbose']:
        print("📁 目錄設置完成")

# Step 1. 影片轉成幀
@time_it("提取幀", CONFIG)
def extract_frames(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ 找不到視頻文件: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("❌ 無法打開視頻文件")
    
    # 獲取視頻資訊
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    if CONFIG['debug']['verbose']:
        print(f"📹 視頻資訊:")
        print(f"   - FPS: {fps:.2f}")
        print(f"   - 總幀數: {total_frames}")
        print(f"   - 時長: {duration:.2f} 秒")
    
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{config.FRAMES_DIR}/frame_{idx:04d}.jpg", frame)
        idx += 1
        
        # 顯示進度
        if CONFIG['debug']['verbose'] and idx % CONFIG['display']['progress_interval'] == 0:
            progress = (idx / total_frames) * 100
            print(f"   進度: {progress:.1f}% ({idx}/{total_frames})")
    
    cap.release()
    if CONFIG['debug']['verbose']:
        print(f"✅ 成功提取 {idx} 幀")
    return fps  # 返回fps供後續使用

# Step 2. SAM 分割
@time_it("SAM 物件分割", CONFIG)
def sam_segment(frame_path, checkpoint):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"❌ 找不到 SAM 模型文件: {checkpoint}")
    
    if CONFIG['debug']['verbose']:
        print("🤖 正在載入 SAM 模型...")
    sam = sam_model_registry[config.MODEL_TYPE](checkpoint=checkpoint)
    if use_cuda:
        sam = sam.cuda()
    predictor = SamPredictor(sam)

    image = cv2.imread(frame_path)
    if image is None:
        raise FileNotFoundError(f"❌ 找不到幀文件: {frame_path}")
    
    predictor.set_image(image)

    # 互動式點選
    click_coords = []
    window_name = config.WINDOW_TITLE

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_coords.append((x, y))
            if CONFIG['debug']['verbose']:
                print(f"🖱️ 點選位置: ({x}, {y})")
            # 在圖像上標記點選位置
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, image)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, image)
    
    if CONFIG['debug']['verbose']:
        print("🖱️ 請在圖像上點選要處理的物件...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not click_coords:
        raise ValueError("❌ 沒有點選任何位置，程式終止")

    input_point = np.array([click_coords[0]])
    input_label = np.array([1])

    if CONFIG['debug']['verbose']:
        print("🔍 正在生成分割遮罩...")
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )

    mask = masks[0]
    
    # 顯示分割結果供確認
    if config.SHOW_PREVIEW:
        masked_image = image.copy()
        masked_image[mask] = masked_image[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
        cv2.imshow("分割結果預覽 (按任意鍵繼續)", masked_image.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    cv2.imwrite(f"{config.MASKS_DIR}/mask_0000.png", mask.astype(np.uint8) * 255)
    if CONFIG['debug']['verbose']:
        print(f"✅ 分割完成，置信度: {scores[0]:.3f}")
    return mask

@time_it("追蹤與遮罩傳播", CONFIG)
def track_and_propagate_mask(initial_mask, num_frames):
    tracker = cv2.TrackerCSRT_create()

    # 讀取初始 frame & 遮罩
    init_frame_path = f"{config.FRAMES_DIR}/frame_0000.jpg"
    frame = cv2.imread(init_frame_path)
    mask = initial_mask.astype(np.uint8) * 255

    # 推測初始遮罩 bounding box
    x, y, w, h = cv2.boundingRect(mask)

    # 初始化追蹤器
    tracker.init(frame, (x, y, w, h))

    for i in range(num_frames):
        frame_path = f"{config.FRAMES_DIR}/frame_{i:04d}.jpg"
        frame = cv2.imread(frame_path)

        success, box = tracker.update(frame)
        if not success:
            print(f"⚠️ 第{i}幀追蹤失敗")
            continue

        x, y, w, h = [int(v) for v in box]

        # 將初始遮罩貼到新位置
        new_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        mask_crop = mask[y:y+h, x:x+w]
        new_mask[y:y+h, x:x+w] = mask_crop

        out_path = f"{config.MASKS_DIR}/mask_{i:04d}.png"
        cv2.imwrite(out_path, new_mask)
        
        if CONFIG['debug']['verbose'] and i % 30 == 0:
            print(f"   幀 {i}: 追蹤位置 ({x}, {y}, {w}, {h})")

    print("✅ 遮罩追蹤與傳播完成")

# Step 4. 重新著色
@time_it("物件重新著色", CONFIG)
def recolor_object(target_color=None):
    if target_color is None:
        target_color = config.TARGET_COLOR
    
    frame_files = sorted([f for f in os.listdir(config.FRAMES_DIR) if f.endswith('.jpg')])
    total_frames = len(frame_files)
    
    if CONFIG['debug']['verbose']:
        print(f"🎨 正在重新著色 {total_frames} 幀...")
    
    for idx, fname in enumerate(frame_files):
        frame_path = os.path.join(config.FRAMES_DIR, fname)
        mask_path = f"{config.MASKS_DIR}/mask_{idx:04d}.png"
        
        if not os.path.exists(mask_path):
            if CONFIG['debug']['verbose']:
                print(f"⚠️ 警告: 找不到遮罩文件 {mask_path}")
            continue
            
        frame = cv2.imread(frame_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if frame is None or mask is None:
            if CONFIG['debug']['verbose']:
                print(f"⚠️ 警告: 無法讀取文件 - frame: {frame_path}, mask: {mask_path}")
            continue
            
        mask_binary = mask > 128
        
        # 應用新顏色
        frame[mask_binary] = target_color
        
        output_path = f"{config.RECOLORED_DIR}/frame_{idx:04d}.jpg"
        cv2.imwrite(output_path, frame)
        
        # 顯示進度
        if CONFIG['debug']['verbose'] and (idx + 1) % CONFIG['display']['progress_interval'] == 0:
            progress = ((idx + 1) / total_frames) * 100
            print(f"   進度: {progress:.1f}% ({idx + 1}/{total_frames})")
    
    if CONFIG['debug']['verbose']:
        print("✅ 重新著色完成")

# Step 5. 合成視頻
@time_it("合成輸出視頻", CONFIG)
def build_video_from_frames(fps=30):
    frame_files = sorted([f for f in os.listdir(config.RECOLORED_DIR) if f.endswith('.jpg')])
    
    if not frame_files:
        raise FileNotFoundError("❌ 找不到重新著色的幀文件")

    first_frame = cv2.imread(os.path.join(config.RECOLORED_DIR, frame_files[0]))
    if first_frame is None:
        raise ValueError("❌ 無法讀取第一幀")
    
    h, w, _ = first_frame.shape
    if CONFIG['debug']['verbose']:
        print(f"📹 輸出視頻規格: {w}x{h} @ {fps}fps")

    # 使用配置中的編碼器
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
    out = cv2.VideoWriter(config.OUTPUT_VIDEO, fourcc, fps, (w, h))
    
    total_frames = len(frame_files)
    for idx, frame_name in enumerate(frame_files):
        frame_path = os.path.join(config.RECOLORED_DIR, frame_name)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            if CONFIG['debug']['verbose']:
                print(f"⚠️ 警告: 無法讀取幀 {frame_path}")
            continue
            
        out.write(frame)
        
        # 顯示進度
        if CONFIG['debug']['verbose'] and (idx + 1) % CONFIG['display']['progress_interval'] == 0:
            progress = ((idx + 1) / total_frames) * 100
            print(f"   進度: {progress:.1f}% ({idx + 1}/{total_frames})")
    
    out.release()
    if CONFIG['debug']['verbose']:
        print(f"✅ 輸出視頻已保存: {config.OUTPUT_VIDEO}")

# 🚀 主流程
@time_it("整體處理", CONFIG)
def main():
    try:
        if CONFIG['debug']['verbose']:
            print(f"🚀 開始處理視頻: {config.VIDEO_PATH}")
            print(f"⏰ 開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 設置目錄
        setup_directories()
        
        # 提取幀
        fps = extract_frames(config.VIDEO_PATH)
        
        # SAM 分割
        mask = sam_segment(f"{config.FRAMES_DIR}/frame_0000.jpg", config.CHECKPOINT_PATH)
        
        # 遮罩傳播
        num_frames = len([f for f in os.listdir(config.FRAMES_DIR) if f.endswith('.jpg')])
        track_and_propagate_mask(mask, num_frames)
        
        # 重新著色
        recolor_object()
        
        # 合成視頻 - 使用原始FPS或預設值
        output_fps = fps if CONFIG['output']['preserve_fps'] else 30
        build_video_from_frames(output_fps)
        
        if CONFIG['debug']['verbose']:
            print(f"🎉 所有處理完成！")
            print(f"⏰ 結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ 發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    
# import cv2
# print(hasattr(cv2, 'TrackerCSRT_create'))  # 應該會輸出 True
