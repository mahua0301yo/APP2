import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import cv2
import numpy as np
import av

# 初始化 YOLO 模型
model = YOLO('yolov8n-pose.pt')  # 確保 yolov8n-pose.pt 文件存在於當前目錄

# 自定義處理器，用來處理攝像頭流並進行 YOLO 偵測
class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        # 將 frame 轉換為 numpy array
        img = frame.to_ndarray(format="bgr24")

        # 使用 YOLO 模型進行偵測
        results = model(img)

        # 獲取偵測後的影像結果
        result_img = np.squeeze(results.render())

        # 將處理後的影像轉換回 frame 返回
        return av.VideoFrame.from_ndarray(result_img, format="bgr24")

# Streamlit 應用
st.title("手機瀏覽器即時鏡頭物件偵測")

# 啟動 WebRTC 並運行 YOLO 偵測
webrtc_streamer(key="yolo", video_processor_factory=YOLOProcessor)

st.write("正在使用 YOLO 模型進行即時物件偵測。請允許使用攝像頭。")
