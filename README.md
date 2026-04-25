# 🛡️ AI Threat Detection & Surveillance System

An intelligent real-time surveillance system that uses Deep Learning models to detect threats such as weapons, fire, smoke, and violent activity from video streams.

---

## 🚀 Features

- 🔫 Weapon Detection (Gun, Knife) using YOLO
- 🔥 Fire & Smoke Detection using YOLO
- 🥊 Violence Detection using CNN
- 🎥 Supports:
  - Video file upload
  - Live camera stream (via link)
- 🚨 Real-time alert generation
- 📊 Detection analytics and metrics
- 📁 CSV export of detection logs (timestamp-based)
- 🖥️ Interactive UI built with Streamlit

---

## 🧠 Models Used

| Model Type | Purpose |
|-----------|--------|
| YOLO | Gun & Knife Detection |
| YOLO | Fire & Smoke Detection |
| CNN | Violence Detection |

---

## 🖥️ User Interface

Built using **Streamlit**, allowing users to:

- Upload videos
- Provide live camera links
- Run detection in real-time
- View alerts and analytics

---

## 📊 Output

- Real-time detection alerts
- Summary metrics:
  - Total alerts
  - Type of threats detected
- Downloadable CSV file with:
  - Timestamp
  - Detection type
  - Confidence score

---

## 🛠️ Tech Stack

- Python
- YOLO (Object Detection)
- CNN (Violence Classification)
- OpenCV
- Streamlit
- Pandas

---

## 📂 Project Structure
├── models/
├── app.py
├── utils/
├── outputs/
├── requirements.txt
└── README.md


---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/repo-name.git

# Navigate to project
cd repo-name

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py


---

If you want, I can also:
- Make your GitHub description (1–2 lines)
- Add badges (accuracy, Python version, etc.)
- Help you write a strong resume bullet for this project

Just tell me 👍
