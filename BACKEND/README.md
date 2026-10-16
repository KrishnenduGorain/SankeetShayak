# Project HandShayak: Physics-Informed Sign-Language Recognition 

> **HandShayak** (Hand Helper) is a physics-informed AI system that models the human hand as a biomechanical rigid-body linkage to improve sign-language recognition.

## 📖 Overview
Conventional sign-language recognition systems rely almost entirely on visual handshape and landmark trajectories. However, visually similar signs (e.g., a "gentle tap" vs. a "forceful strike") can have vastly different movement characteristics. 

Project **HandShayak** goes beyond static imagery by treating the hand as a mechanical system. Using **MediaPipe**, we extract 3D landmarks and compute physical dynamic features—such as velocity, acceleration, jerk, kinetic energy, and joint torque—to give our machine learning classifiers a deeper understanding of *how* the hand moves, not just *what* it looks like.

## ✨ Key Features
* **Real-time Landmark Detection:** Powered by Google's MediaPipe Hands.
* **Physics & Kinematics Engine:** Calculates linear/angular velocity, acceleration, and jerk using Savitzky–Golay smoothed trajectories.
* **Biomechanical Dynamics:** Approximates segment mass and moment of inertia to compute kinetic energy, momentum, and inferred net force.
* **Ablation Study Framework:** Designed to objectively compare shape-only, dynamics-only, and combined feature classifiers.
* **Real-Time API:** FastAPI & WebSocket integration for low-latency, continuous video stream prediction (coming soon).

## 🛠️ Tech Stack
* **Computer Vision:** OpenCV, MediaPipe
* **Physics & Math:** NumPy, SciPy (Savitzky-Golay filters)
* **Machine Learning:** Scikit-Learn (SVM, Random Forest), ONNX Runtime
* **Backend API:** FastAPI, WebSockets
* **DevOps:** Docker, GitHub Actions

## 🧑‍💻 The Team & Roles

* **Monojoy:** Frontend Lead (Client-Side UI, WebSockets & Visualization)
* H**arsimran:** ML & Backend Architect (FastAPI, Model Training & ONNX Export)
* **Krishnendu:** DevOps Lead (Dockerization, Cloud Deployment & CI/CD)
* **Jai Krishna:** Physics Engine Developer (Kinematics, Dynamics & Math via NumPy)
* **Debonnita:** Vision Pipeline Developer (MediaPipe 3D Landmark Extraction)
* **Koushik:** Data Engineer & QA (Feature Aggregation, Pytest Unit Testing)

## 🚀 Installation & Setup

*(Note: The codebase is currently under active development based on our production roadmap. Setup instructions will be updated as modules are completed.)*

### Prerequisites
* Python 3.9+
* Recommended: Virtual Environment (`venv` or `conda`)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/hand-shayak.git
cd hand-shayak

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 📊 System Architecture
1. **Video Input:** Captures real-time webcam frames.
2. **MediaPipe:** Extracts 21 3D hand coordinates.
3. **Kinematics Engine:** Smooths data and calculates $v$, $a$, $jerk$.
4. **Dynamics Engine:** Calculates kinetic energy ($KE = \frac{1}{2}mv^2 + \frac{1}{2}I\omega^2$) and torque.
5. **Feature Aggregation:** Combines shape vectors and dynamic peaks.
6. **Inference Server:** ML model predicts the sign label.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
