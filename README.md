# SafeBrowse: Advanced AI-Powered Parental Control

SafeBrowse is a next-generation parental control application designed to provide real-time, intelligent protection for children online. Unlike traditional filters that rely solely on blacklists, SafeBrowse uses **local AI models** to analyze content (text and images) in real-time, detecting harmful material such as pornography, violence, gore, hate speech, and self-harm content with high accuracy.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/backend-FastAPI-green.svg)
![React Native](https://img.shields.io/badge/frontend-Expo-blue.svg)
![AI](https://img.shields.io/badge/AI-PyTorch%20%2B%20Transformers-orange.svg)

## 🚀 Key Features

### 🛡️ Smart Protection
- **Multi-Model AI Analysis**: Uses an ensemble of AI models (Falconsai, AdamCodd, Toxic-Comment) for robust detection.
- **Real-Time Image Filtering**: Analyzes images on the fly using GPU acceleration (if available) to block NSFW, gore, and violent imagery.
- **Context-Aware Text Analysis**: Detects bullying, hate speech, sexual content, and self-harm triggers in text.
- **Expanded Safety Categories**: Blocks not just adult content, but also:
  - 🩸 Gore & Violence
  - 💊 Drugs & Substance Abuse
  - 🎲 Gambling
  - 🔪 Self-Harm & Suicide
  - 🤬 Hate Speech & Toxic Language
  - 🚫 Age-Restricted Social Media (Users < 16 blocked from FB, Insta, TikTok, etc.)

### 👨‍👩‍👧‍👦 Parent Dashboard
- **Profile Management**: Create custom profiles for each child with age-appropriate sensitivity levels (Strict, Moderate, Lenient).
- **Activity Monitoring**: View detailed logs of blocked content, complete with reasoning (e.g., "AI detected violence with 95% confidence").
- **Secure Access**: PIN-protected parent mode ensures children cannot tamper with settings.
- **Searchable Logs**: Easily filter and search through activity history.
- **Wellbeing Insights**: Get weekly digests of browsing habits, safety scores, and top blocked categories without needing to read every log.

### 📱 Integrated Child Browser
- **Safe Webview**: A custom built-in browser that filters content before it renders.
- **Visual Feedback**: Clear "Page Blocked" screens when harmful content is detected, educating the child on why it was blocked.
- **Seamless Experience**: Fast browsing with minimal latency thanks to optimized local inference.

---

## 🛠️ Tech Stack

### Backend (AI & API)
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Database**: [MongoDB](https://www.mongodb.com/) (Async Motor Driver)
- **AI/ML**: 
  - [PyTorch](https://pytorch.org/) & [Hugging Face Transformers](https://huggingface.co/)
  - Models: `Falconsai/nsfw_image_detection`, `AdamCodd/vit-base-nsfw-detector`, `martin-ha/toxic-comment-model`
  - **GPU Support**: Automatic CUDA detection for high-performance inference.
- **Authentication**: JWT (JSON Web Tokens) with Bcrypt hashing.

### Frontend (Mobile App)
- **Framework**: [Expo](https://expo.dev/) (React Native 0.79)
- **Navigation**: Expo Router (File-based routing)
- **State Management**: Zustand & React Context
- **UI Components**: Native components with custom styling & animations
- **Browser Engine**: `react-native-webview` with injected JavaScript for content extraction.

---

## 📂 Project Structure

```bash
Safe-Browse/
├── backend/
│   ├── server.py              # Main FastAPI application & AI Logic
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Backend configuration
│
├── frontend/
│   ├── app/                   # Expo Router screens
│   │   ├── auth/              # Login/Signup screens
│   │   ├── parent/            # Dashboard & Settings
│   │   └── child/             # Safe Browser
│   ├── components/            # Reusable UI components
│   ├── contexts/              # Global state (Auth, Mode)
│   ├── assets/                # Images & fonts
│   └── package.json           # Node dependencies
```

---

## ⚡ Getting Started

### Prerequisites
- **Node.js** (v18+) & **Yarn**
- **Python** (v3.10+)
- **MongoDB** (Installed and running locally on port 27017)
- **Expo Go** app on your mobile device (Android/iOS)

### 1. Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: This installs PyTorch. If you have a supported NVIDIA GPU, ensure you install the CUDA-enabled version of PyTorch manually for best performance.)*

4. Configure Environment Variables:
   Create a `.env` file in `backend/` and add:
   ```env
   MONGO_URL=mongodb://localhost:27017
   DB_NAME=safebrowse_db
   JWT_SECRET_KEY=your_super_secret_key_here
   ```

5. Start the Server:
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8001 --reload
   ```
   You should see logs indicating that AI models are loading (this may take a minute on first run).

### 2. Frontend Setup

1. Open a new terminal and navigate to the frontend:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   yarn install
   ```

3. Start the Expo development server:
   ```bash
   yarn start
   ```

4. **Run on Device**: Scan the QR code displayed in the terminal using the **Expo Go** app on your phone. Ensure your phone and computer are on the **same Wi-Fi network**.

---

## 📖 Usage Guide

### Parent Setup
1. **Sign Up**: Create an account when you open the app.
2. **Set PIN**: Go to Settings and configure a 4-digit security PIN. This checks prevents children from exiting Safe Mode.
3. **Create Profile**: Go to the Profiles tab and add a profile for your child (e.g., "Alice, Age 10").
   - **Strict (5-8)**: High sensitivity, strict filters.
   - **Moderate (9-12)**: Balanced protection.
   - **Lenient (13+)**: Blocks only severe content.

### Safe Browsing (Child Mode)
1. Select a child profile from the dashboard to enter **Child Mode**.
2. The device is now locked into the Safe Browser.
3. **Filtering**: As the child browses, text and images are analyzed. If they attempt to visit a blocked site or view harmful content, a "Screen Blocked" overlay appears.
4. **Exit**: To leave Child Mode, tap the exit button and enter your Parent PIN.

---

## 📡 API Endpoints

### Auth
- `POST /api/auth/signup`: Create a new parent account.
- `POST /api/auth/login`: Authenticate and get JWT.

### Profiles
- `GET /api/profiles`: List all child profiles.
- `POST /api/profiles`: Create a new profile.
- `PUT /api/profiles/{id}`: Update specific profile settings.

### Analysis & Logs
- `POST /api/content/analyze`: Submit text/image/url for safety check.
- `GET /api/logs`: Retrieve activity history for profiles.
- `GET /api/insights/{profile_id}`: Retrieve aggregated wellbeing stats and insights.

---

## 🤝 Contributing

Contributions are welcome! This is an active project aiming to make the internet safer for kids.

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

## 📄 License

This project is licensed under the **MIT License**.
