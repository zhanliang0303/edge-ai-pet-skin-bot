import os
import io
import json
import asyncio
import sqlite3
from datetime import datetime
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)

# 載入 .env 檔案中的環境變數
load_dotenv()

# ==========================================
# 🛠️ 全域參數設定
# ==========================================
TOKEN = os.getenv("TELEGRAM_TOKEN")
MODEL_PATH = "docker_best.onnx"
KB_PATH = "knowledge_base.json"
DB_PATH = "diagnoses.db"
IMAGES_DIR = "saved_images"

# 模型分類標籤 (順序必須與 ONNX 模型輸出一致)
CLASS_KEYS = [
    "bacterial dermatosis",
    "fungal infection",
    "healthy",
    "hypersensitivity dermatitis"
]

# ===== 系統超參數 =====
CONFIDENCE_THRESHOLD = 0.80     # 高信心直接回覆的閾值
ENTROPY_THRESHOLD = 0.70        # 熵超過此值視為灰色地帶
OOD_ENERGY_THRESHOLD = 0.65     # OOD Score (1-max_p) > 此值視為 OOD (相當於 max_p < 0.35)
MAX_QUESTIONS = 3               # 問診最多問幾題
CONVERGENCE_THRESHOLD = 0.92    # Bayesian 收斂閾值 (提高至 0.92，防止灰色地帶或外部病症過早收斂)
DEFAULT_SYMPTOM_PROB = 0.15     # 症狀不在某疾病列表中時的預設 P(symptom|disease)

# ==========================================
# 🗄️ SQLite 資料庫設定與輔助函式
# ==========================================

def init_db():
    """初始化 SQLite 資料庫與圖片儲存目錄"""
    if not os.path.exists(IMAGES_DIR):
        try:
            os.makedirs(IMAGES_DIR)
            print(f"📁 已建立圖片儲存目錄：{IMAGES_DIR}")
        except Exception as e:
            print(f"⚠️ 無法建立圖片儲存目錄: {e}")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diagnosis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                chat_id INTEGER,
                timestamp TEXT,
                image_path TEXT,
                initial_top_class TEXT,
                initial_top_prob REAL,
                initial_entropy REAL,
                initial_energy REAL,
                route_taken TEXT,
                questions_asked INTEGER,
                final_diagnosis TEXT,
                final_probability REAL,
                comorbidity_detected INTEGER,
                second_diagnosis TEXT,
                second_probability REAL
            )
        """)
        conn.commit()
        conn.close()
        print(f"🗄️ SQLite 資料庫初始化成功：{DB_PATH}")
    except Exception as e:
        print(f"⚠️ 初始化資料庫失敗: {e}")


def log_diagnosis_to_db(
    user_id: int | None,
    username: str | None,
    chat_id: int | None,
    image_path: str | None,
    initial_top_class: str | None,
    initial_top_prob: float | None,
    initial_entropy: float | None,
    initial_energy: float | None,
    route_taken: str,
    questions_asked: int,
    final_diagnosis: str | None,
    final_probability: float | None,
    comorbidity_detected: int = 0,
    second_diagnosis: str | None = None,
    second_probability: float = 0.0
):
    """將診斷紀錄寫入 SQLite 資料庫"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO diagnosis_history (
                user_id, username, chat_id, timestamp, image_path,
                initial_top_class, initial_top_prob, initial_entropy, initial_energy,
                route_taken, questions_asked, final_diagnosis, final_probability,
                comorbidity_detected, second_diagnosis, second_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, username, chat_id, timestamp, image_path,
            initial_top_class, initial_top_prob, initial_entropy, initial_energy,
            route_taken, questions_asked, final_diagnosis, final_probability,
            comorbidity_detected, second_diagnosis, second_probability
        ))
        conn.commit()
        conn.close()
        print(f"💾 已成功寫入一筆診斷紀錄到資料庫 (Route: {route_taken}, User: {username})")
    except Exception as e:
        print(f"⚠️ 寫入資料庫失敗: {e}")


# ==========================================
# 🚀 載入 ONNX 推論引擎
# ==========================================
print(f"🚀 初始化 ONNX 推論引擎 ({MODEL_PATH})...")
options = ort.SessionOptions()
options.intra_op_num_threads = 1
try:
    session = ort.InferenceSession(
        MODEL_PATH, sess_options=options, providers=['CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name
    print("✅ 模型載入成功")
except Exception as e:
    print(f"⚠️ 載入模型失敗，請確認檔案是否存在: {e}")
    session = None

# ==========================================
# 📚 載入知識庫
# ==========================================
print(f"📚 載入知識庫 ({KB_PATH})...")
try:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        KNOWLEDGE_BASE = json.load(f)
    print(f"✅ 知識庫載入成功（{len(KNOWLEDGE_BASE['diseases'])} 種分類）")
except Exception as e:
    print(f"⚠️ 載入知識庫失敗: {e}")
    KNOWLEDGE_BASE = None


# ==========================================
# 🔢 數學工具函式
# ==========================================

def compute_softmax(logits: np.ndarray) -> np.ndarray:
    """
    計算 Softmax 機率分佈（含數值穩定性處理）
    σ(z_i) = exp(z_i - max(z)) / Σ exp(z_j - max(z))
    """
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


def compute_entropy(probs: np.ndarray) -> float:
    """
    計算 Shannon Entropy（預測不確定性）
    H(p) = -Σ p_i × ln(p_i)

    - H = 0      → 完全確定（只有一類機率為 1）
    - H = ln(4)  → 完全不確定（四類均等）≈ 1.386
    """
    p = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(p * np.log(p)))


def compute_energy_score(logits: np.ndarray, temperature: float = 1.0) -> float:
    """
    計算 Energy Score 用於 OOD 偵測
    E(x) = -T × log(Σ exp(f_i(x) / T))

    In-distribution → energy 較低（更負，如 -8）
    OOD             → energy 較高（接近 0 或正值）

    參考：Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020

    使用 LogSumExp trick 確保數值穩定
    """
    scaled = logits / temperature
    max_val = np.max(scaled)
    return float(-temperature * (max_val + np.log(np.sum(np.exp(scaled - max_val)))))


# ==========================================
# 🖼️ 圖片預處理
# ==========================================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    圖片預處理流程：
    1. 開啟圖片並轉換為 RGB
    2. 縮放為 224x224（與訓練一致）
    3. 正規化 (0~1) + HWC→CHW 轉換
    4. 擴充 Batch 維度 → (1, 3, 224, 224)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))      # HWC → CHW
    return np.expand_dims(img_array, axis=0)              # 加 Batch 維度


def detect_human_face(image_bytes: bytes) -> bool:
    """
    使用 OpenCV Haar Cascade 偵測人臉，避免將人臉誤判為健康寵物皮膚
    """
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,   # 提高相鄰矩形閾值，減少對狗狗皮膚褶皺、陰影或手部的誤判
            minSize=(60, 60)  # 提高人臉最小尺寸限制，過濾主動上傳的清晰人臉即可
        )
        return len(faces) > 0
    except Exception as e:
        print(f"⚠️ 人臉偵測發生錯誤: {e}")
        return False


# ==========================================
# 🧠 推論引擎（升級版）
# ==========================================

async def run_inference(input_tensor: np.ndarray) -> dict | None:
    """
    執行 ONNX 推論，回傳完整分析結果：
    - softmax 機率向量 (直接從模型輸出讀取，不重複套用 softmax)
    - 預測熵 H(p)
    - OOD Score (使用 MSP 方法 1.0 - max(p))
    """
    if session is None:
        return None

    outputs = await asyncio.to_thread(session.run, None, {input_name: input_tensor})
    probs = outputs[0][0] # ONNX 模型已輸出 Softmax 機率分布

    entropy = compute_entropy(probs)
    # 使用 MSP (Maximum Softmax Probability) 量化 OOD 程度：1.0 - max(p)
    ood_score = float(1.0 - np.max(probs))

    top_idx = int(np.argmax(probs))

    return {
        "logits": probs, # 保留相容性
        "probabilities": probs,
        "entropy": entropy,
        "energy_score": ood_score, # 變數名稱維持 energy_score 以相容舊代碼
        "top_class": CLASS_KEYS[top_idx],
        "top_confidence": float(probs[top_idx]),
        "all_probs": {CLASS_KEYS[i]: float(probs[i]) for i in range(len(CLASS_KEYS))}
    }


# ==========================================
# 🩺 Bayesian 問診引擎
# ==========================================

class BayesianEngine:
    """
    基於貝葉斯推理的多輪問診引擎

    原理：
    1. 以 YOLO 的 softmax 輸出作為先驗機率 P(Disease)
    2. 每次用戶回答症狀問題後，以 knowledge_base.json 中的
       條件機率 P(Symptom | Disease) 進行 Bayesian Update：

       P(D|S) ∝ P(S|D) × P(D)

    3. 重複直到 max(P) ≥ 收斂閾值 或達到最大問題數
    """

    def __init__(self, prior_probs: dict, kb_diseases: dict):
        """
        Args:
            prior_probs: {disease_key: probability} 來自 YOLO softmax
            kb_diseases: KNOWLEDGE_BASE["diseases"] 字典
        """
        self.kb = kb_diseases

        # 保留所有機率 >= 5% 的候選類別（包含 healthy）
        self.candidates = [
            k for k in prior_probs
            if prior_probs[k] >= 0.05
        ]

        # 在候選類別中重新正規化先驗機率
        total = sum(prior_probs[d] for d in self.candidates)
        if total > 0:
            self.posteriors = {d: prior_probs[d] / total for d in self.candidates}
        else:
            n = len(self.candidates)
            self.posteriors = {d: 1.0 / n for d in self.candidates}

        self.asked_questions = set()   # 已問過的問題文字
        self.question_count = 0
        self.current_symptom_key = None  # 當前等待回答的症狀 key

    def _get_symptom_prob(self, disease_key: str, symptom_key: str) -> float:
        """取得 P(symptom | disease)，若為健康類別回傳極低機率，若不存在回傳預設小值"""
        if disease_key == "healthy":
            return 0.05  # 健康寵物出現上述皮膚病理病徵的機率極低
        return self.kb.get(disease_key, {}).get("symptoms", {}).get(
            symptom_key, DEFAULT_SYMPTOM_PROB
        )

    def select_next_question(self) -> str | None:
        """
        選擇 Information Gain 最高的下一個問題

        IG(Q) = H(current) - E[H(after_answer)]
             = H(current) - [P(yes)×H(yes) + P(no)×H(no)]

        Returns:
            問題文字，或 None（無更多問題可問）
        """
        best_question = None
        best_gain = -1.0
        best_symptom = None

        for disease_key in self.candidates:
            info = self.kb.get(disease_key, {})
            questions = info.get("questions", [])
            symptom_keys = list(info.get("symptoms", {}).keys())

            for i, q_text in enumerate(questions):
                if q_text in self.asked_questions:
                    continue
                if i >= len(symptom_keys):
                    continue

                s_key = symptom_keys[i]
                gain = self._compute_info_gain(s_key)

                if gain > best_gain:
                    best_gain = gain
                    best_question = q_text
                    best_symptom = s_key

        if best_question is not None:
            self.current_symptom_key = best_symptom
            self.asked_questions.add(best_question)
            return best_question

        return None

    def _compute_info_gain(self, symptom_key: str) -> float:
        """計算某個症狀問題的 Information Gain"""
        probs_arr = np.array([self.posteriors[d] for d in self.candidates])
        h_before = compute_entropy(probs_arr)

        # P(answer=yes) = Σ_d P(d) × P(symptom | d)
        p_yes = sum(
            self.posteriors[d] * self._get_symptom_prob(d, symptom_key)
            for d in self.candidates
        )
        p_yes = np.clip(p_yes, 0.01, 0.99)
        p_no = 1.0 - p_yes

        # 模擬回答「是」後的後驗分佈
        post_yes = {
            d: self.posteriors[d] * self._get_symptom_prob(d, symptom_key)
            for d in self.candidates
        }
        t = sum(post_yes.values())
        if t > 0:
            post_yes = {k: v / t for k, v in post_yes.items()}
        h_yes = compute_entropy(np.array(list(post_yes.values())))

        # 模擬回答「否」後的後驗分佈
        post_no = {
            d: self.posteriors[d] * (1.0 - self._get_symptom_prob(d, symptom_key))
            for d in self.candidates
        }
        t = sum(post_no.values())
        if t > 0:
            post_no = {k: v / t for k, v in post_no.items()}
        h_no = compute_entropy(np.array(list(post_no.values())))

        # IG = H(before) - E[H(after)]
        return h_before - (p_yes * h_yes + p_no * h_no)

    def update(self, answer: str):
        """
        根據用戶回答更新後驗機率

        Bayesian Update:
            P(D | answer=yes) ∝ P(symptom | D) × P(D)
            P(D | answer=no)  ∝ (1 - P(symptom | D)) × P(D)
            answer=unsure     → 不更新（likelihood = 1）

        Args:
            answer: "yes" / "no" / "unsure"
        """
        if self.current_symptom_key is None:
            return

        self.question_count += 1

        if answer == "unsure":
            self.current_symptom_key = None
            return

        for d in self.candidates:
            p = self._get_symptom_prob(d, self.current_symptom_key)
            if answer == "yes":
                self.posteriors[d] *= p
            else:  # "no"
                self.posteriors[d] *= (1.0 - p)

        # 正規化
        total = sum(self.posteriors[d] for d in self.candidates)
        if total > 0:
            for d in self.candidates:
                self.posteriors[d] /= total

        self.current_symptom_key = None

    def is_converged(self) -> bool:
        """判斷是否收斂：最高機率 ≥ 閾值 或 已問滿最大題數"""
        if not self.candidates:
            return True
        max_p = max(self.posteriors[d] for d in self.candidates)
        return max_p >= CONVERGENCE_THRESHOLD or self.question_count >= MAX_QUESTIONS

    def get_result(self) -> dict | None:
        """取得最終診斷結果，包含共病偵測"""
        if not self.candidates:
            return None

        # 按機率排序
        ranked = sorted(self.candidates, key=lambda d: self.posteriors[d], reverse=True)
        top_d = ranked[0]
        top_p = self.posteriors[top_d]
        info = self.kb.get(top_d, {})

        # 共病偵測：第二名機率仍 > 25%（且第一名不能為健康，第二名也不能是健康）
        comorbidity = False
        second_d = None
        if top_d != "healthy" and len(ranked) >= 2:
            second_d = ranked[1]
            if self.posteriors[second_d] > 0.25 and second_d != "healthy":
                comorbidity = True

        return {
            "disease_key": top_d,
            "name_zh": info.get("name_zh", top_d),
            "name_en": info.get("name_en", top_d),
            "probability": top_p,
            "advice": info.get("advice", "建議就醫"),
            "description": info.get("description", ""),
            "urgency": info.get("urgency", "medium"),
            "converged": top_p >= CONVERGENCE_THRESHOLD,
            "comorbidity": comorbidity,
            "second_key": second_d if comorbidity else None,
            "second_name_zh": self.kb.get(second_d, {}).get("name_zh", "") if comorbidity else None,
            "second_prob": self.posteriors[second_d] if comorbidity and second_d else 0,
            "questions_asked": self.question_count,
            "all_posteriors": dict(self.posteriors)
        }


# ==========================================
# 📱 Telegram Bot Handlers
# ==========================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理 /start 指令"""
    context.user_data.clear()
    welcome = (
        "你好！我是大白 🤖\n"
        "你的專屬寵物皮膚健康 AI 助手。\n\n"
        "📸 請傳送一張寵物皮膚狀況的照片，我來幫你初步評估！\n\n"
        "我的能力：\n"
        "🔬 辨識細菌性皮膚病、黴菌感染、過敏性皮膚炎\n"
        "🩺 不確定時會透過問診進一步判斷\n"
        "⚠️ 遇到無法辨識的狀況會提醒您就醫\n\n"
        "_(請注意：本分析僅供參考，無法取代專業獸醫診斷)_"
    )
    await update.message.reply_text(welcome, parse_mode="Markdown")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    處理使用者傳送的圖片 —— 核心三路分流邏輯
    儲存影像至本地，並將診斷結果寫入 SQLite 資料庫
    """
    # 重置任何進行中的問診
    context.user_data.clear()

    user_id = update.effective_user.id if update.effective_user else 0
    username = update.effective_user.username if update.effective_user else "Unknown"
    chat_id = update.effective_chat.id

    msg = await update.message.reply_text("🔍 大白正在分析這張照片，請稍候...")

    try:
        # 下載圖片
        photo_file = await update.message.photo[-1].get_file()
        img_bytes = await photo_file.download_as_bytearray()

        # 儲存照片到本地
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"{user_id}_{timestamp_str}.jpg"
        local_img_path = os.path.join(IMAGES_DIR, img_filename)
        
        try:
            with open(local_img_path, "wb") as img_f:
                img_f.write(img_bytes)
        except Exception as e_save:
            print(f"⚠️ 無法儲存圖片到本地: {e_save}")
            local_img_path = None

        # 1. 執行人臉過濾器，防止人臉被診斷為健康寵物
        if detect_human_face(img_bytes):
            ood_msg = (
                "⚠️ **大白警示：偵測到人臉**\n\n"
                "本系統專為**寵物（狗、貓）皮膚病**健康輔診設計，無法用於診斷人類皮膚狀況，亦不可替代人類醫療診斷。\n\n"
                "若您上傳的是寵物照片但被誤判，請確保照片中**只有寵物皮膚患部**（避免人臉或背景入鏡），並請重新上傳照片。"
            )
            
            # 寫入資料庫
            log_diagnosis_to_db(
                user_id=user_id,
                username=username,
                chat_id=chat_id,
                image_path=local_img_path,
                initial_top_class="OOD_human_face",
                initial_top_prob=1.0,
                initial_entropy=0.0,
                initial_energy=1.0,
                route_taken="ood",
                questions_asked=0,
                final_diagnosis="OOD / Human Face Detected",
                final_probability=1.0,
                comorbidity_detected=0
            )
            
            await msg.edit_text(ood_msg, parse_mode="Markdown")
            return

        # 預處理 + 推論
        tensor = preprocess_image(img_bytes)
        result = await run_inference(tensor)

        if result is None:
            await msg.edit_text("❌ 系統未載入模型，請聯繫管理員。")
            return

        top_class = result["top_class"]
        top_conf = result["top_confidence"]
        entropy = result["entropy"]
        energy = result["energy_score"]
        all_probs = result["all_probs"]

        # ============================================================
        # Route 1: 高信心 → 直接回覆或進入小類別
        # ============================================================
        if top_conf >= CONFIDENCE_THRESHOLD and entropy < ENTROPY_THRESHOLD:
            if top_class != "healthy" and KNOWLEDGE_BASE:
                context.user_data["user_id"] = user_id
                context.user_data["username"] = username
                context.user_data["chat_id"] = chat_id
                context.user_data["image_path"] = local_img_path
                context.user_data["initial_top_class"] = top_class
                context.user_data["initial_top_prob"] = top_conf
                context.user_data["initial_entropy"] = entropy
                context.user_data["initial_energy"] = energy
                context.user_data["route_taken"] = "high_confidence"
                
                await msg.delete()
                await _start_subcategory_diagnosis(update, context, top_class)
                return

            disease_info = KNOWLEDGE_BASE["diseases"].get(top_class, {}) if KNOWLEDGE_BASE else {}
            name_zh = disease_info.get("name_zh", top_class)
            advice = disease_info.get("advice", "建議就醫")
            description = disease_info.get("description", "")
            urgency = disease_info.get("urgency", "medium")
            urgency_icon = {"high": "🔴", "medium": "🟡", "none": "🟢"}.get(urgency, "⚪")

            text = (
                f"✅ **分析完成！**\n\n"
                f"🩺 **初步評估**：`{name_zh}`\n"
                f"📊 **AI 信心指數**：`{top_conf * 100:.1f}%`\n"
                f"{urgency_icon} **緊急程度**：`{urgency}`\n\n"
                f"📝 **說明**：{description}\n\n"
                f"💡 **建議**：{advice}\n\n"
                f"📈 _預測熵 H = {entropy:.3f} | Energy = {energy:.2f}_\n"
                f"_(本分析僅供參考，無法取代專業獸醫診斷)_"
            )
            
            log_diagnosis_to_db(
                user_id=user_id,
                username=username,
                chat_id=chat_id,
                image_path=local_img_path,
                initial_top_class=top_class,
                initial_top_prob=top_conf,
                initial_entropy=entropy,
                initial_energy=energy,
                route_taken="high_confidence",
                questions_asked=0,
                final_diagnosis=top_class,
                final_probability=top_conf,
                comorbidity_detected=0
            )

            await msg.edit_text(text, parse_mode="Markdown")
            return

        # ============================================================
        # Route 2: OOD 偵測 → 拒絕診斷
        # ============================================================
        if energy > OOD_ENERGY_THRESHOLD:
            ood_msg = KNOWLEDGE_BASE.get("ood_message", "⚠️ 無法辨識") if KNOWLEDGE_BASE else "⚠️ 無法辨識"
            text = (
                f"🔍 **分析完成**\n\n"
                f"📊 模型最高信心度：`{top_conf * 100:.1f}%`\n"
                f"📈 不確定性指標 H = `{entropy:.3f}`\n"
                f"🔋 OOD Energy Score = `{energy:.3f}`\n\n"
                f"{ood_msg}"
            )
            
            # 寫入資料庫
            log_diagnosis_to_db(
                user_id=user_id,
                username=username,
                chat_id=chat_id,
                image_path=local_img_path,
                initial_top_class=top_class,
                initial_top_prob=top_conf,
                initial_entropy=entropy,
                initial_energy=energy,
                route_taken="ood",
                questions_asked=0,
                final_diagnosis="OOD / Unknown",
                final_probability=1.0,
                comorbidity_detected=0
            )

            await msg.edit_text(text, parse_mode="Markdown")
            return

        # ============================================================
        # Route 3: 灰色地帶 → 啟動 Bayesian 問診
        # ============================================================
        if KNOWLEDGE_BASE is None:
            await msg.edit_text("❌ 知識庫未載入，無法啟動問診功能。")
            return

        # 初始化 Bayesian 引擎
        engine = BayesianEngine(all_probs, KNOWLEDGE_BASE["diseases"])
        context.user_data["engine"] = engine
        context.user_data["user_id"] = user_id
        context.user_data["username"] = username
        context.user_data["chat_id"] = chat_id
        context.user_data["image_path"] = local_img_path
        context.user_data["initial_top_class"] = top_class
        context.user_data["initial_top_prob"] = top_conf
        context.user_data["initial_entropy"] = entropy
        context.user_data["initial_energy"] = energy

        if top_class == "healthy":
            # 啟動「健康預篩選」機制
            context.user_data["state"] = "healthy_prescreening"
            init_text = (
                f"🔍 **初步分析完成**\n\n"
                f"AI 傾向認為寵物看起來可能健康，但信心度尚未達到 100%（目前為 `{top_conf * 100:.1f}%`）。\n"
                f"為確保準確度，大白需要進行簡單的症狀確認。"
            )
            await msg.edit_text(init_text, parse_mode="Markdown")
            await asyncio.sleep(1.0)

            # 發送預篩選問題
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("❌ 否，沒有任何症狀", callback_data="prescreen_no_symptoms"),
                    InlineKeyboardButton("✅ 是，有出現症狀", callback_data="prescreen_has_symptoms"),
                ]
            ])
            text = (
                f"🩺 **健康確認**\n\n"
                f"您的寵物目前有出現任何皮膚異常症狀嗎？\n"
                f"（例如：脫毛、異常皮屑、紅疹、紅腫、傷口、或是頻繁抓撓等）"
            )
            await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=keyboard,
                parse_mode="Markdown"
            )
        else:
            context.user_data["state"] = "consulting"
            
            # 顯示初步分析結果
            probs_lines = []
            for k, v in sorted(all_probs.items(), key=lambda x: -x[1]):
                if k == "healthy":
                    continue
                d_name = KNOWLEDGE_BASE["diseases"].get(k, {}).get("name_zh", k)
                bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
                probs_lines.append(f"  {bar} {v * 100:.1f}% {d_name}")
            probs_text = "\n".join(probs_lines)

            init_text = (
                f"🔍 **初步分析完成**\n\n"
                f"AI 對多個可能的狀況信心接近，需要進一步了解症狀。\n\n"
                f"📊 **各項機率分佈**：\n{probs_text}\n\n"
                f"📈 _預測熵 H = {entropy:.3f}（灰色地帶）_\n\n"
                f"🩺 大白將透過最多 {MAX_QUESTIONS} 個問題來進一步判斷..."
            )
            await msg.edit_text(init_text, parse_mode="Markdown")

            # 短暫停頓後開始問第一個問題（UX 體驗）
            await asyncio.sleep(1.0)
            await _ask_next_question(update, context)

    except Exception as e:
        print(f"❌ 處理圖片時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        await msg.edit_text("❌ 抱歉，大白在分析圖片時遇到了問題，請換一張照片再試一次。")


async def _ask_next_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """從 BayesianEngine 取得下一個問題並發送"""
    engine: BayesianEngine = context.user_data.get("engine")
    if engine is None:
        return

    question = engine.select_next_question()
    if question is None:
        # 沒有更多問題可問，直接輸出結果
        await _send_diagnosis(update, context)
        return

    q_num = engine.question_count + 1

    # 建立 InlineKeyboard 按鈕
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ 是", callback_data="answer_yes"),
            InlineKeyboardButton("❌ 否", callback_data="answer_no"),
            InlineKeyboardButton("🤷 不確定", callback_data="answer_unsure"),
        ]
    ])

    text = f"🩺 **問診 ({q_num}/{MAX_QUESTIONS})**\n\n{question}"

    chat_id = update.effective_chat.id
    await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=keyboard,
        parse_mode="Markdown"
    )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理 InlineKeyboard 按鈕回調（是/否/不確定 及 健康預篩選）"""
    query = update.callback_query
    await query.answer()

    state = context.user_data.get("state")
    
    if state == "sub_consulting":
        sub_engine: BayesianEngine = context.user_data.get("sub_engine")
        if sub_engine is None:
            await query.edit_message_text("⚠️ 問診已結束。")
            return
            
        answer_map = {
            "sub_answer_yes": ("yes", "✅ 是"),
            "sub_answer_no": ("no", "❌ 否"),
            "sub_answer_unsure": ("unsure", "🤷 不確定"),
        }
        answer, answer_label = answer_map.get(query.data, ("unsure", "🤷 不確定"))

        original_text = query.message.text
        await query.edit_message_text(f"{original_text}\n\n👉 你的回答：{answer_label}")

        sub_engine.update(answer)

        if sub_engine.is_converged():
            await asyncio.sleep(0.5)
            await _send_final_subcategory_diagnosis(update, context)
        else:
            await asyncio.sleep(0.5)
            await _ask_next_subcategory_question(update, context)
        return

    engine: BayesianEngine = context.user_data.get("engine")

    if state not in ["consulting", "healthy_prescreening"] or engine is None:
        await query.edit_message_text("⚠️ 問診已結束，請重新傳送照片開始新的分析。")
        return

    # 處理健康預篩選狀態
    if state == "healthy_prescreening":
        if query.data == "prescreen_no_symptoms":
            # 顯示回答、移除按鈕
            await query.edit_message_text(f"{query.message.text}\n\n👉 你的回答：❌ 否，沒有任何症狀")
            
            # 用戶宣稱完全無症狀，直接將 healthy 機率設為 1.0，其餘設為 0.0
            for d in engine.candidates:
                if d == "healthy":
                    engine.posteriors[d] = 1.0
                else:
                    engine.posteriors[d] = 0.0
            
            # 直接進入診斷發送
            await asyncio.sleep(0.5)
            await _send_diagnosis(update, context)
            return

        elif query.data == "prescreen_has_symptoms":
            await query.edit_message_text(f"{query.message.text}\n\n👉 你的回答：✅ 是，有出現症狀")
            
            # 用戶表示有症狀，將 healthy 的機率歸零，並把權重重新分配給其他候選疾病
            if "healthy" in engine.candidates:
                engine.posteriors["healthy"] = 0.0
                others = [c for c in engine.candidates if c != "healthy"]
                if others:
                    total_p = sum(engine.posteriors[o] for o in others)
                    if total_p > 0:
                        for o in others:
                            engine.posteriors[o] /= total_p
                    else:
                        for o in others:
                            engine.posteriors[o] = 1.0 / len(others)
                else:
                    # 如果候選名單只有 healthy，但用戶又說有症狀，則放回所有疾病
                    engine.candidates = [c for c in CLASS_KEYS if c != "healthy"]
                    for c in engine.candidates:
                        engine.posteriors[c] = 1.0 / len(engine.candidates)
            
            # 切換狀態為 normal consulting，並開始發問
            context.user_data["state"] = "consulting"
            
            # 顯示目前更新後的機率分佈
            post_lines = []
            for d in sorted(engine.candidates, key=lambda x: engine.posteriors[x], reverse=True):
                if d == "healthy":
                    continue
                d_name = KNOWLEDGE_BASE["diseases"].get(d, {}).get("name_zh", d)
                p = engine.posteriors[d]
                bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
                post_lines.append(f"  {bar} {p * 100:.1f}% {d_name}")
            
            chat_id = update.effective_chat.id
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"📊 _目前機率更新（已排除健康選項）：_\n" + "\n".join(post_lines),
                parse_mode="Markdown"
            )
            
            await asyncio.sleep(0.5)
            await _ask_next_question(update, context)
            return

    # 原有的普通問診流程
    # 解析回答
    answer_map = {
        "answer_yes": ("yes", "✅ 是"),
        "answer_no": ("no", "❌ 否"),
        "answer_unsure": ("unsure", "🤷 不確定"),
    }
    answer, answer_label = answer_map.get(query.data, ("unsure", "🤷 不確定"))

    # 更新訊息：顯示回答、移除按鈕
    original_text = query.message.text
    await query.edit_message_text(f"{original_text}\n\n👉 你的回答：{answer_label}")

    # Bayesian 更新
    engine.update(answer)

    # 顯示即時機率更新（讓用戶看到機率在變化）
    post_lines = []
    for d in sorted(engine.candidates, key=lambda x: engine.posteriors[x], reverse=True):
        d_name = KNOWLEDGE_BASE["diseases"].get(d, {}).get("name_zh", d)
        p = engine.posteriors[d]
        bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
        post_lines.append(f"  {bar} {p * 100:.1f}% {d_name}")

    chat_id = update.effective_chat.id
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"📊 _目前機率更新：_\n" + "\n".join(post_lines),
        parse_mode="Markdown"
    )

    # 判斷是否收斂
    if engine.is_converged():
        await asyncio.sleep(0.5)
        await _send_diagnosis(update, context)
    else:
        await asyncio.sleep(0.5)
        await _ask_next_question(update, context)


async def _send_diagnosis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """輸出大類別問診結果，若需細分則進入小類別問診"""
    engine: BayesianEngine = context.user_data.get("engine")
    if engine is None:
        return

    result = engine.get_result()
    chat_id = context.user_data.get("chat_id") or (update.effective_chat.id if update.effective_chat else 0)

    if result is None:
        if chat_id:
            await context.bot.send_message(chat_id=chat_id, text="❌ 無法取得大類別診斷結果。")
        context.user_data.clear()
        return

    # 若大類別收斂，且不是健康，則進入小類別問診
    if result["converged"] and result["disease_key"] != "healthy":
        context.user_data["macro_result"] = result
        await _start_subcategory_diagnosis(update, context, result["disease_key"])
        return

    # 如果沒有收斂，或是健康，則直接輸出並寫入資料庫
    user_id = context.user_data.get("user_id")
    username = context.user_data.get("username")
    image_path = context.user_data.get("image_path")
    initial_top_class = context.user_data.get("initial_top_class")
    initial_top_prob = context.user_data.get("initial_top_prob")
    initial_entropy = context.user_data.get("initial_entropy")
    initial_energy = context.user_data.get("initial_energy")
    route_taken = context.user_data.get("route_taken", "bayesian_consultation")
    
    context.user_data.clear()

    urgency_icon = {"high": "🔴", "medium": "🟡", "none": "🟢"}.get(result["urgency"], "⚪")

    post_lines = []
    for k, v in sorted(result["all_posteriors"].items(), key=lambda x: -x[1]):
        d_name = KNOWLEDGE_BASE["diseases"].get(k, {}).get("name_zh", k)
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        post_lines.append(f"  {bar} {v * 100:.1f}% {d_name}")
    posteriors_text = "\n".join(post_lines)

    status = "✅ 問診收斂" if result["converged"] else "⚠️ 無法確定具體病症"

    text = (
        f"🏥 **問診完成！**\n\n"
        f"{status}（共問了 {result['questions_asked']} 題）\n\n"
        f"🩺 **最可能狀況**：`{result['name_zh']}` (`{result.get('name_en', result['disease_key'])}`)\n"
        f"📊 **問診後信心**：`{result['probability'] * 100:.1f}%`\n"
        f"{urgency_icon} **緊急程度**：`{result['urgency']}`\n\n"
    )

    if result["comorbidity"]:
        text += (
            f"⚠️ **共病提醒**：`{result['second_name_zh']}` "
            f"的機率仍有 `{result['second_prob'] * 100:.1f}%`，\n"
            f"寵物可能同時存在多種皮膚問題（如過敏繼發細菌感染）。\n\n"
        )

    if not result["converged"]:
        text += "💡 **建議**：症狀較為複雜，建議盡快帶至獸醫院進行完整皮膚檢查。\n\n"
    else:
        text += f"💡 **建議**：{result['advice']}\n\n"

    text += (
        f"📈 **各疾病後驗機率**：\n{posteriors_text}\n\n"
        f"_(本分析僅供參考，無法取代專業獸醫診斷)_"
    )

    if chat_id:
        await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")

    log_diagnosis_to_db(
        user_id=user_id,
        username=username,
        chat_id=chat_id,
        image_path=image_path,
        initial_top_class=initial_top_class,
        initial_top_prob=initial_top_prob,
        initial_entropy=initial_entropy,
        initial_energy=initial_energy,
        route_taken=route_taken,
        questions_asked=result["questions_asked"],
        final_diagnosis=result["disease_key"],
        final_probability=result["probability"],
        comorbidity_detected=1 if result["comorbidity"] else 0,
        second_diagnosis=result["second_key"] if result["comorbidity"] else None,
        second_probability=result["second_prob"] if result["comorbidity"] else 0.0
    )


async def _start_subcategory_diagnosis(update: Update, context: ContextTypes.DEFAULT_TYPE, macro_class: str):
    """啟動小類別問診"""
    subcategories = KNOWLEDGE_BASE["diseases"].get(macro_class, {}).get("subcategories", {})
    if not subcategories:
        context.user_data.clear()
        if update.effective_chat:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="分析完成（無小類別資訊）。")
        return

    n = len(subcategories)
    priors = {k: 1.0 / n for k in subcategories.keys()}
    sub_engine = BayesianEngine(priors, subcategories)
    
    context.user_data["state"] = "sub_consulting"
    context.user_data["sub_engine"] = sub_engine
    context.user_data["macro_class"] = macro_class
    
    macro_name = KNOWLEDGE_BASE["diseases"].get(macro_class, {}).get("name_zh", macro_class)
    chat_id = context.user_data.get("chat_id") or (update.effective_chat.id if update.effective_chat else 0)
    
    msg = (
        f"✅ **初步分析大類別為：{macro_name}**\n\n"
        f"為了給予更精確的用藥與照護建議，大白需要再問您幾個細節問題以判斷小類別..."
    )
    if chat_id:
        await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
        
    await asyncio.sleep(1.0)
    await _ask_next_subcategory_question(update, context)


async def _ask_next_subcategory_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """從小類別問診引擎取得下一個問題"""
    engine: BayesianEngine = context.user_data.get("sub_engine")
    if engine is None:
        return

    question = engine.select_next_question()
    if question is None:
        await _send_final_subcategory_diagnosis(update, context)
        return

    q_num = engine.question_count + 1
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ 是", callback_data="sub_answer_yes"),
            InlineKeyboardButton("❌ 否", callback_data="sub_answer_no"),
            InlineKeyboardButton("🤷 不確定", callback_data="sub_answer_unsure"),
        ]
    ])

    text = f"🔬 **細分問診 ({q_num}/{MAX_QUESTIONS})**\n\n{question}"
    chat_id = context.user_data.get("chat_id") or (update.effective_chat.id if update.effective_chat else 0)
    
    if chat_id:
        await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=keyboard, parse_mode="Markdown")


async def _send_final_subcategory_diagnosis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """輸出最終包含大類別與小類別的診斷結果"""
    sub_engine: BayesianEngine = context.user_data.get("sub_engine")
    macro_class = context.user_data.get("macro_class")
    macro_name = KNOWLEDGE_BASE["diseases"].get(macro_class, {}).get("name_zh", macro_class)
    macro_en = KNOWLEDGE_BASE["diseases"].get(macro_class, {}).get("name_en", macro_class)
    
    user_id = context.user_data.get("user_id")
    username = context.user_data.get("username")
    chat_id = context.user_data.get("chat_id") or (update.effective_chat.id if update.effective_chat else 0)
    image_path = context.user_data.get("image_path")
    initial_top_class = context.user_data.get("initial_top_class")
    initial_top_prob = context.user_data.get("initial_top_prob")
    initial_entropy = context.user_data.get("initial_entropy")
    initial_energy = context.user_data.get("initial_energy")
    route_taken = context.user_data.get("route_taken", "bayesian_consultation")
    macro_result = context.user_data.get("macro_result")

    sub_result = sub_engine.get_result()
    total_questions = sub_result["questions_asked"]
    if macro_result:
        total_questions += macro_result.get("questions_asked", 0)

    context.user_data.clear()

    if not sub_result:
        if chat_id:
            await context.bot.send_message(chat_id=chat_id, text="❌ 無法取得細分結果。")
        return

    urgency_icon = {"high": "🔴", "medium": "🟡", "none": "🟢"}.get(sub_result["urgency"], "⚪")
    
    final_diagnosis_str = f"{macro_class} ({sub_result['disease_key']})"

    text = (
        f"✅ **分析完成！**\n\n"
        f"📋 **大類別**：{macro_name} (`{macro_en}`)\n"
        f"🔬 **細分類型**：{sub_result['name_zh']} (`{sub_result.get('name_en', sub_result['disease_key'])}`)\n"
        f"📊 **細分信心**：`{sub_result['probability'] * 100:.1f}%`\n"
        f"{urgency_icon} **緊急程度**：`{sub_result['urgency'].upper()}`\n\n"
        f"💊 **就醫指引**：\n{sub_result['advice']}\n\n"
        f"_(本分析僅供參考，無法取代專業獸醫診斷)_"
    )

    if chat_id:
        await context.bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")

    log_diagnosis_to_db(
        user_id=user_id,
        username=username,
        chat_id=chat_id,
        image_path=image_path,
        initial_top_class=initial_top_class,
        initial_top_prob=initial_top_prob,
        initial_entropy=initial_entropy,
        initial_energy=initial_energy,
        route_taken=route_taken,
        questions_asked=total_questions,
        final_diagnosis=final_diagnosis_str,
        final_probability=sub_result["probability"],
        comorbidity_detected=1 if (macro_result and macro_result.get("comorbidity")) else 0,
        second_diagnosis=macro_result.get("second_key") if macro_result else None,
        second_probability=macro_result.get("second_prob") if macro_result else 0.0
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """處理純文字訊息"""
    state = context.user_data.get("state")

    if state == "consulting":
        await update.message.reply_text(
            "🩺 目前正在問診中，請使用上方的按鈕回答問題。\n"
            "如果要重新開始，請傳送新的照片或輸入 /start。"
        )
    else:
        await update.message.reply_text(
            "📸 請傳送一張寵物皮膚狀況的照片，我來幫你評估！\n"
            "或輸入 /start 查看使用說明。"
        )


# ==========================================
# 🚀 主程式
# ==========================================

def main():
    if not TOKEN or TOKEN == "YOUR_TELEGRAM_TOKEN_HERE" or TOKEN == "YOUR_TELEGRAM_TOKEN":
        print("⚠️ 請先在 .env 檔案中填入你的 Telegram Bot TOKEN (TELEGRAM_TOKEN)！")
        return

    # 初始化 SQLite 資料庫與相片目錄
    init_db()

    if KNOWLEDGE_BASE is None:
        print("⚠️ 知識庫未載入，問診功能將無法使用。")

    print("🤖 正在啟動大白 RAG 診斷 Agent...")

    app = ApplicationBuilder().token(TOKEN).build()

    # 路由註冊（順序很重要）
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(CallbackQueryHandler(handle_callback))  # 問診按鈕回調
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("✅ 大白 RAG 診斷 Agent 已上線！等待接收訊息...")
    print(f"   📊 信心閾值: {CONFIDENCE_THRESHOLD}")
    print(f"   📈 熵閾值: {ENTROPY_THRESHOLD}")
    print(f"   🔋 OOD 閾值: {OOD_ENERGY_THRESHOLD}")
    print(f"   🩺 最大問診題數: {MAX_QUESTIONS}")
    app.run_polling()


if __name__ == "__main__":
    main()
