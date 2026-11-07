import io
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

# 頁面基本設定
st.set_page_config(page_title="影像處理期中專題系統", layout="wide")
st.title("數位影像處理與人臉分析互動系統")

# ==========================================
# 輔助函式庫 (Utility Functions)
# ==========================================


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """計算兩張影像之間的 Peak Signal-to-Noise Ratio (PSNR)。"""
    if img1.shape != img2.shape:
        return 0.0
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def parse_kernel_input(kernel_text: str, size: int) -> np.ndarray:
    """解析使用者輸入的字串為二維卷積核。"""
    try:
        rows = [r.strip() for r in kernel_text.strip().split("\n") if r.strip()]
        if len(rows) != size:
            return None
        matrix = []
        for r in rows:
            values = [float(x) for x in r.replace(",", " ").split() if x]
            if len(values) != size:
                return None
            matrix.append(values)
        return np.array(matrix, dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None


# ==========================================
# 側邊欄：功能選單與影像載入
# ==========================================

st.sidebar.header("功能導覽與控制面板")
uploaded_file = st.sidebar.file_uploader(
    "上傳影像 (支援 PNG, JPG, JPEG, BMP)", type=["png", "jpg", "jpeg", "bmp"]
)

tab_choice = st.sidebar.radio(
    "選擇功能模組：",
    [
        "1. 影像載入與中繼資訊",
        "2. 色彩空間轉換與直方圖分析",
        "3. 自訂卷積濾波與 PSNR 評估",
        "4. 人臉偵測模組 (MediaPipe)",
    ],
)

if uploaded_file is not None:
    # 讀取影像檔案
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    file_size_kb = len(file_bytes) / 1024.0
    height, width, channels = img_rgb.shape

    # ------------------------------------------
    # 模組 1: 影像載入與基礎解析
    # ------------------------------------------
    if tab_choice == "1. 影像載入與中繼資訊":
        st.header("模組 1：影像載入與基礎解析")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(img_rgb, caption="原始輸入影像", use_container_width=True)
        with col2:
            st.subheader("影像中繼資訊 (Metadata)")
            st.markdown(f"- **檔案名稱**：`{uploaded_file.name}`")
            st.markdown(f"- **檔案大小**：`{file_size_kb:.2f} KB`")
            st.markdown(f"- **解析度 (寬 × 高)**：`{width} × {height}` 像素")
            st.markdown(f"- **色彩維度 (Channels)**：`{channels}` 通道 (RGB)")
            st.markdown(f"- **資料型態 (Dtype)**：`{img_rgb.dtype}`")

    # ------------------------------------------
    # 模組 2: 多色彩空間轉換與同步直方圖分析
    # ------------------------------------------
    elif tab_choice == "2. 色彩空間轉換與直方圖分析":
        st.header("模組 2：多色彩空間轉換與同步直方圖分析")

        mode = st.selectbox(
            "切換檢視色彩通道模式：",
            [
                "Original (全彩)",
                "R 通道",
                "G 通道",
                "B 通道",
                "Y 通道 (亮度)",
                "U 通道 (色度)",
                "V 通道 (色度)",
            ],
        )

        r_ch, g_ch, b_ch = cv2.split(img_rgb)
        img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        y_ch, u_ch, v_ch = cv2.split(img_yuv)

        display_img = img_rgb
        hist_data = None
        hist_color = "black"

        if mode == "Original (全彩)":
            display_img = img_rgb
        elif mode == "R 通道":
            display_img = r_ch
            hist_data = r_ch
            hist_color = "red"
        elif mode == "G 通道":
            display_img = g_ch
            hist_data = g_ch
            hist_color = "green"
        elif mode == "B 通道":
            display_img = b_ch
            hist_data = b_ch
            hist_color = "blue"
        elif mode == "Y 通道 (亮度)":
            display_img = y_ch
            hist_data = y_ch
            hist_color = "gray"
        elif mode == "U 通道 (色度)":
            display_img = u_ch
            hist_data = u_ch
            hist_color = "purple"
        elif mode == "V 通道 (色度)":
            display_img = v_ch
            hist_data = v_ch
            hist_color = "orange"

        col_img, col_hist = st.columns(2)
        with col_img:
            st.subheader(f"通道顯示：{mode}")
            st.image(display_img, use_container_width=True, clamp=True)

        with col_hist:
            st.subheader("同步像素強度直方圖")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            if mode == "Original (全彩)":
                for ch, color, name in zip(
                    [r_ch, g_ch, b_ch], ["red", "green", "blue"], ["R", "G", "B"]
                ):
                    hist, _ = np.histogram(ch, bins=256, range=[0, 256])
                    ax.plot(hist, color=color, label=f"{name} 通道", alpha=0.8)
                ax.legend()
            else:
                hist, _ = np.histogram(hist_data, bins=256, range=[0, 256])
                ax.plot(hist, color=hist_color, lw=1.5)

            ax.set_xlim([0, 255])
            ax.set_xlabel("像素強度 (0-255)")
            ax.set_ylabel("像素數量")
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)
            plt.close(fig)

    # ------------------------------------------
    # 模組 3: 自訂卷積核濾波與 PSNR 評估
    # ------------------------------------------
    elif tab_choice == "3. 自訂卷積濾波與 PSNR 評估":
        st.header("模組 3：自訂卷積核空間濾波與量化評估 (PSNR)")

        col_cfg, col_res = st.columns([1, 2])
        with col_cfg:
            k_size = st.radio("選擇卷積核維度：", [3, 5], horizontal=True)
            preset = st.selectbox(
                "快速預設範本：",
                [
                    "自訂 (Custom)",
                    "高斯平滑 (Gaussian)",
                    "邊緣銳化 (Sharpen)",
                    "索伯邊緣 (Sobel Horizontal)",
                    "浮雕效果 (Emboss)",
                ],
            )

            # 預設矩陣定義
            default_3x3 = "0 0 0\n0 1 0\n0 0 0"
            default_5x5 = "0 0 0 0 0\n0 0 0 0 0\n0 0 1 0 0\n0 0 0 0 0\n0 0 0 0 0"

            if k_size == 3:
                if preset == "高斯平滑 (Gaussian)":
                    default_3x3 = "1 2 1\n2 4 2\n1 2 1"
                elif preset == "邊緣銳化 (Sharpen)":
                    default_3x3 = "0 -1 0\n-1 5 -1\n0 -1 0"
                elif preset == "索伯邊緣 (Sobel Horizontal)":
                    default_3x3 = "-1 -2 -1\n0 0 0\n1 2 1"
                elif preset == "浮雕效果 (Emboss)":
                    default_3x3 = "-2 -1 0\n-1 1 1\n0 1 2"
                kernel_text = st.text_area(
                    "輸入 3×3 矩陣 (每列換行，數值以空格隔開)：",
                    value=default_3x3,
                    height=120,
                )
            else:
                if preset == "高斯平滑 (Gaussian)":
                    default_5x5 = (
                        "1 4 6 4 1\n4 16 24 16 4\n6 24 36 24 6\n4 16 24 16 4\n1 4 6 4 1"
                    )
                kernel_text = st.text_area(
                    "輸入 5×5 矩陣 (每列換行，數值以空格隔開)：",
                    value=default_5x5,
                    height=150,
                )

            auto_normalize = st.checkbox("自動正規化 (除以矩陣元素總和)", value=False)

        kernel = parse_kernel_input(kernel_text, k_size)

        with col_res:
            if kernel is None:
                st.error(f"矩陣格式錯誤！請確認輸入為 {k_size}×{k_size} 的數值矩陣。")
            else:
                k_sum = np.sum(kernel)
                if auto_normalize and k_sum != 0:
                    kernel = kernel / k_sum

                # 執行卷積空間濾波運算 (邊界模式鏡射反射 BORDER_REFLECT)
                filtered_rgb = cv2.filter2D(
                    img_rgb, -1, kernel, borderType=cv2.BORDER_REFLECT
                )
                # 截斷確保數值在 0-255 範圍
                filtered_rgb = np.clip(filtered_rgb, 0, 255).astype(np.uint8)

                # 計算 PSNR
                psnr_val = calculate_psnr(img_rgb, filtered_rgb)

                st.subheader(
                    f"濾波品質指標：PSNR = `{psnr_val:.2f} dB`"
                    if psnr_val != float("inf")
                    else "濾波品質指標：PSNR = `∞ (影像無失真)`"
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.image(img_rgb, caption="原始影像", use_container_width=True)
                with c2:
                    st.image(
                        filtered_rgb,
                        caption=f"濾波後影像 ({preset})",
                        use_container_width=True,
                    )

    # ------------------------------------------
    # 模組 4: 人臉偵測模組 (OpenCV Haar-Cascade)
    # ------------------------------------------
    elif tab_choice == "4. 人臉偵測模組 (MediaPipe)":
        st.header("模組 4：經典特徵辨識人臉偵測模組 (Haar-Cascade)")

        # 參數微調滑桿
        scale_factor = st.slider("尺度縮放比例 (scaleFactor)", 1.05, 1.5, 1.1, 0.05)
        min_neighbors = st.slider("最少相鄰矩形數 (minNeighbors)", 1, 10, 5, 1)

        annotated_img = img_rgb.copy()
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # 1. 確保 Haar-Cascade XML 模型檔案存在（本地找不到則自動下載）
        import os
        import urllib.request

        xml_filename = "haarcascade_frontalface_default.xml"
        cascade_path = (
            os.path.join(cv2.data.haarcascades, xml_filename)
            if hasattr(cv2, "data")
            else xml_filename
        )

        # 若標準路徑不存在該檔案，則從官方 GitHub 鏡像下載至當前目錄快取
        if not os.path.exists(cascade_path):
            cascade_path = os.path.join(os.getcwd(), xml_filename)
            if not os.path.exists(cascade_path):
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                with st.spinner("首次執行正在下載 Haar 人臉模型權重檔..."):
                    urllib.request.urlretrieve(url, cascade_path)

        # 2. 載入分類器
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            st.error(
                "無法正確載入 Haar-Cascade 權重檔案，請確認網路連線或手動下載 XML。"
            )
            st.stop()

        # 3. 執行人臉偵測 (detectMultiScale3 取得邊界框與信心權重)
        faces, rejectLevels, levelWeights = face_cascade.detectMultiScale3(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30),
            outputRejectLevels=True,
        )

        face_count = len(faces)
        detections_info = []

        if face_count > 0:
            for idx, ((x, y, w, h), weight) in enumerate(zip(faces, levelWeights)):
                # 將權重轉換為標準化信心度分數 (Sigmoid 映射)
                conf = float(1.0 / (1.0 + np.exp(-weight / 10.0)))

                # 繪製邊界框與標籤
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"#{idx + 1}: {conf:.2f}"
                cv2.putText(
                    annotated_img,
                    label,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                detections_info.append(
                    {
                        "人臉編號": idx + 1,
                        "信心度 (Confidence)": f"{conf:.4f}",
                        "邊界座標 (x, y, w, h)": f"({x}, {y}, {w}, {h})",
                    }
                )

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(
                annotated_img,
                caption=f"人臉標註成果 (共偵測到 {face_count} 張人臉)",
                use_container_width=True,
            )
        with col2:
            st.subheader("偵測統計結果")
            st.metric("偵測到人臉總數", f"{face_count} 人")
            if detections_info:
                st.table(detections_info)
            else:
                st.info(
                    "未偵測到任何人臉，請嘗試調降 minNeighbors 或微調 scaleFactor。"
                )
