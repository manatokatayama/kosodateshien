# app.py
import os
import streamlit as st
import cv2
from PIL import Image
import torch
from face_detect import run_face_detection, select_emoji
from mozaikushori import load_model_s, load_model_m, load_model_l
from glob import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_s = load_model_s("yolov5s-face.pt", device)
model_m = load_model_m("yolov5m-face.pt", device)
model_l = load_model_l("yolov5l-face.pt", device)

st.title("匿名化アプリ（絵文字＆モザイク対応）")

mode_choice = st.radio("処理モードを選択", ["emoji", "mosaic"])

emoji_path = None
emoji_category = None
emoji_scale = 1.8
sequential_mode = False  # 🟢 初期値

if mode_choice == "emoji":
    st.subheader("🎨 絵文字設定")

    emoji_mode = st.radio("絵文字の選択方法", ["カテゴリ指定", "単体指定"])

    if emoji_mode == "カテゴリ指定":
        emoji_category = st.selectbox("カテゴリを選択", ["ハート", "春", "水色", "動物"])
        emoji_category = select_emoji(emoji_category, category_mode=True)

        # 🟢 新オプション: 貼り付け順序選択
        order_mode = st.radio("貼り付け順序を選択", ["ランダム", "順番"])
        if order_mode == "順番":
            sequential_mode = True

    else:
        emoji_choice = st.selectbox(
            "単体絵文字を選択してください",
            [
                "ハート(黄色)", "ハート(白)",
                "春(エッグ)", "春(桜)", "春(桜２)", "春(梅)",
                "水色(アイス)", "水色(クローバー)", "水色(スペード)",
                "水色(ハート)", "水色(マカロン)",
                "動物(キリン)", "動物(タヌキ)", "動物(パンダ)",
                "動物(ライオン)", "動物(リス)"
            ]
        )
        emoji_path = select_emoji(emoji_choice, category_mode=False)
        if os.path.exists(emoji_path):
            st.image(emoji_path, caption=f"選択中: {emoji_choice}", width=100)
        else:
            st.warning("⚠ 絵文字画像が見つかりません。")

    emoji_scale = st.slider("スタンプサイズ倍率", 0.5, 3.0, 1.8, 0.1)

elif mode_choice == "mosaic":
    st.subheader("🟫 モザイク設定")
    st.info("※現在はモザイク強度固定です")

uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])
output_folder = "C:/kosodateshien3/output"
os.makedirs(output_folder, exist_ok=True)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    temp_input_folder = "C:/kosodateshien3/temp_input"
    os.makedirs(temp_input_folder, exist_ok=True)
    temp_img_path = os.path.join(temp_input_folder, uploaded_file.name)
    img.save(temp_img_path)

    progress_text = st.empty()
    progress_text.text("🔄 画像処理中...")
    progress_bar = st.progress(0)

    with st.spinner("顔検出と処理を実行中..."):
        result_folder_path = run_face_detection(
            model_s, model_m, model_l,
            input_folder=temp_input_folder,
            device=device,
            output_folder=output_folder,
            mode=mode_choice,
            progress_bar=progress_bar,
            emoji_path=emoji_path,
            emoji_category=emoji_category,
            emoji_scale=emoji_scale,
            sequential_mode=sequential_mode  # 🟢 追加
        )

    progress_text.text("✅ 完了しました！")
    result_img_path = os.path.join(result_folder_path, uploaded_file.name)
    if os.path.exists(result_img_path):
        result_img = Image.open(result_img_path)
        st.image(result_img, caption="処理後の画像", use_container_width=True)
        st.success(f"完了！ 保存先: {result_img_path}")
    else:
        st.error("❌ 処理に失敗しました。")

# ===========================
# 絵文字ギャラリー
# ===========================
st.markdown("---")
if st.button("📁 登録済み絵文字一覧を表示"):
    emoji_base = "C:/kosodateshien3/emoji"
    if os.path.exists(emoji_base):
        subfolders = [f for f in os.listdir(emoji_base) if os.path.isdir(os.path.join(emoji_base, f))]
        st.subheader("登録カテゴリ一覧")
        for sub in subfolders:
            st.markdown(f"### 📂 {sub}")
            paths = glob(os.path.join(emoji_base, sub, "*.png"))
            cols = st.columns(5)
            for i, path in enumerate(paths):
                with cols[i % 5]:
                    st.image(path, width=80)
    else:
        st.error("⚠ emojiフォルダが見つかりません。")