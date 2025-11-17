#face_detect.py
import os
import cv2
import torch
from mozaikushori import detect_one

# 🟢 絵文字フォルダのベースパス
BASE_DIR = r"C:\kosodateshien3\emoji"

def run_face_detection(
    model_s, model_m, model_l,
    input_folder, device, output_folder,
    emoji_path=None, progress_bar=None,
    mode="emoji", emoji_category=None,
    emoji_scale=1.8, sequential_mode=False  # 🟢 順番モード対応
):
    """
    顔検出を行い、絵文字またはモザイクを適用する関数。
    
    Args:
        model_s, model_m, model_l: YOLOv5-Face モデル（3サイズ）
        input_folder (str): 入力画像フォルダ
        device (torch.device): 処理デバイス（CPU or GPU）
        output_folder (str): 出力先フォルダ
        emoji_path (str): 絵文字ファイルパス（単一絵文字モード用）
        progress_bar (streamlit.Progress or None): Streamlit 進捗バー
        mode (str): "emoji" または "mosaic"
        emoji_category (str): カテゴリ指定（例："smile", "animal"）
        emoji_scale (float): 絵文字拡大率
        sequential_mode (bool): 左から順番に処理するモード
    """
    
    # 出力フォルダ作成
    os.makedirs(output_folder, exist_ok=True)

    # 入力フォルダ内の画像一覧を取得
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    total = len(image_files)
    if total == 0:
        print("⚠️ 入力フォルダに画像がありません。")
        return

    # 🟢 絵文字カテゴリフォルダの指定（フォルダが存在しない場合は警告）
    emoji_folder = None
    if emoji_category is not None and not emoji_path:
        emoji_folder = os.path.join(BASE_DIR, emoji_category)
        if not os.path.exists(emoji_folder):
            print(f"⚠️ 絵文字カテゴリフォルダが見つかりません: {emoji_folder}")
            emoji_folder = None

    # 各画像を順に処理
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 画像が読み込めません: {img_path}")
            continue

        # 🟢 顔検出＆処理実行
        detect_one(
            model_s=model_s,
            model_m=model_m,
            model_l=model_l,
            im=img,
            device=device,
            emoji_path=emoji_path,
            emoji_folder=emoji_folder,
            save_path=os.path.join(output_folder, img_name),
            mode=mode,
            emoji_scale=emoji_scale,
            sequential_mode=sequential_mode  # 🟢 左から順番処理モードを有効化
        )

        # 🟢 Streamlit の進捗バー更新
        if progress_bar is not None:
            progress = int((i + 1) / total * 100)
            progress_bar.progress(progress)

        print(f"✅ 処理完了: {img_name}")

    print(f"🎉 全 {total} 枚の画像処理が完了しました。出力先: {output_folder}")
    return output_folder


# ---------------------------
# 単体／カテゴリ絵文字選択関数
# ---------------------------
def select_emoji(emoji_choice="ハート", category_mode=False):
    category_dict = {
        "ハート": "ハート",
        "春": "春",
        "水色": "水色",
        "動物": "動物",
    }

    emoji_dict = {
        "ハート(黄色)": os.path.join(BASE_DIR, r"ハート\ハート_黄色.png"),
        "ハート(白)": os.path.join(BASE_DIR, r"ハート\ハート_白.png"),
        "春(エッグ)": os.path.join(BASE_DIR, r"春\春_エッグ.png"),
        "春(桜)": os.path.join(BASE_DIR, r"春\春_桜.png"),
        "春(桜２)": os.path.join(BASE_DIR, r"春\春_桜2.png"),
        "春(梅)": os.path.join(BASE_DIR, r"春\春_梅.png"),
        "水色(アイス)": os.path.join(BASE_DIR, r"水色\水色_アイス.png"),
        "水色(クローバー)": os.path.join(BASE_DIR, r"水色\水色_クローバー.png"),
        "水色(スペード)": os.path.join(BASE_DIR, r"水色\水色_スペード.png"),
        "水色(ハート)": os.path.join(BASE_DIR, r"水色\水色_ハート.png"),
        "水色(マカロン)": os.path.join(BASE_DIR, r"水色\水色_マカロン.png"),
        "動物(キリン)": os.path.join(BASE_DIR, r"動物\動物_キリン.png"),
        "動物(タヌキ)": os.path.join(BASE_DIR, r"動物\動物_タヌキ.png"),
        "動物(パンダ)": os.path.join(BASE_DIR, r"動物\動物_パンダ.png"),
        "動物(ライオン)": os.path.join(BASE_DIR, r"動物\動物_ライオン.png"),
        "動物(リス)": os.path.join(BASE_DIR, r"動物\動物_リス.png"),
    }

    if category_mode:
        folder_name = category_dict.get(emoji_choice, None)
        if folder_name:
            folder_path = os.path.join(BASE_DIR, folder_name)
            if os.path.exists(folder_path):
                return folder_path
        return None
    else:
        return emoji_dict.get(emoji_choice, None)
