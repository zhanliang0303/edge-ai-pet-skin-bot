# 🐶 Bayesian RAD: Edge-AI Pet Skin Disease Diagnostic System

<div align="center">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
  <img src="https://img.shields.io/badge/Award-%F0%9F%A5%88%202nd%20Place-yellow" alt="Award">
  <img src="https://img.shields.io/badge/Framework-YOLO11%20%7C%20ONNX-blue" alt="Framework">
  <img src="https://img.shields.io/badge/Platform-Telegram%20Bot-blue" alt="Platform">
</div>

> **基於邊緣端貝葉斯檢索增強決策之輕量化寵物皮膚病輔助診斷系統**
> 本專案榮獲機器學習課程期末專題競賽 **第二名 🥈** 

---

## 📖 專案介紹 (Introduction)

隨著寵物經濟爆發，寵物醫療資源分佈不均的問題日益嚴重。現有的醫療 AI 多半高度依賴雲端大語言模型 (LLM)，存在推論成本高、無法離線運行，且容易對未知影像產生「過度自信 (Overconfidence)」等致命缺陷。

本專案提出全球首創的 **「無 LLM 依賴之邊緣端 RAD (檢索增強決策) 架構」**。結合極致輕量化的電腦視覺模型與古典貝氏機率學，打造出具備「醫療安全性」與「白盒可解釋性」的邊緣端 Telegram 輔助診斷機器人。

## ✨ 核心亮點 (Key Features)

### 1. 🛡️ OOD 防禦阻斷機制 (Out-of-Distribution Defense)
為堅守「絕不瞎猜 (Do no harm)」的醫療底線，本系統導入 **MSP (Maximum Softmax Probability)** 分佈外偵測演算法。當使用者上傳人臉、風景等非目標影像時，系統會精準攔截並拋出警告，徹底解決傳統 AI 產生的幻覺。

### 2. 🧠 貝氏動態問診 (Bayesian Consultation)
當影像特徵落入模糊地帶，系統不會武斷給出結論，而是觸發基於 **最大資訊增益 (Information Gain)** 的動態問診引擎。系統會向飼主提出 1~3 個關鍵症狀問題（如：是否伴隨惡臭？），並透過貝氏公式即時收斂機率，達成 100% 白盒可解釋性。

### 3. ⚡ 極致邊緣輕量化 (Edge-Native Performance)
透過結構化剪枝與梯度遮罩微調 (Gradient Masking Fine-tuning)，視覺大腦 (YOLO11n-cls) 成功壓縮 **73%**，體積僅餘 **1.55 MB**。在單核心 CPU 環境下即可跑出 **161 FPS** 的超高流暢度，且 5-Fold 平均準確率維持在 **98.74%**。

## ⚙️ 系統架構 (Architecture)

1. **Vision Prior**: YOLO11-cls (ONNX FP16) 提取影像先驗機率。
2. **Safety Gate**: 計算 Shannon Entropy 與 OOD Score。
3. **Three-way Routing**:
   - `Route 1`: 低熵高信心 -> 💡 直接輸出診斷。
   - `Route 2`: 高 OOD 分數 -> 🚫 防禦阻斷並警告。
   - `Route 3`: 高熵灰色地帶 -> 🩺 啟動貝氏動態問診。

## 🛠️ 執行環境 (Tech Stack)
- **Model**: YOLO11-cls, ONNX Runtime
- **Backend**: Python 3.11, python-telegram-bot
- **Deployment**: Docker (Single-core constraint simulation)

## 🏆 專案成就 (Achievements)
- 獲得機器學習期末專題競賽 **全班第二名**。
- 完整通過 OOD 安全性測試與單核 CPU 效能壓測。
- 已封裝為完整可用的 Telegram Bot，具備未來無網域離線原生 App (iOS/Android) 擴充潛力。

---
*Developed by Zhan-Liang Lin & Zong-Xun Guo, 2026.*
