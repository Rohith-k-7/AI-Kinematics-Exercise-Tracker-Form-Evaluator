# AI-Kinematics-Exercise-Tracker-Form-Evaluator


# AI Kinematics Trainer :
### Real-Time Exercise Tracker & Form Evaluator using YOLOv8 Pose Estimation  

AI Kinematics Trainer is a real-time fitness assistant that detects human body posture using **YOLOv8 Pose**, counts repetitions, evaluates form, and provides live feedback with audio coaching.

This project focuses on **Push-ups (Standard & Wide)** and **Sit-ups**, with accurate rep counting and posture correction in real time.

---

##  Features

Real-time pose detection using **YOLOv8 Pose Model**  
Exercise repetition counting  
- Standard Push-ups  
- Wide Push-ups  
- Sit-ups  

Form Evaluation & Posture Feedback  
- Back alignment check  
- Correct push-up support position  
- Sit-up stage detection  

Live Web Dashboard (Flask UI)  
- Total reps  
- Exercise-wise reps  
- Average rep time  
- Feedback display  

Audio Feedback System  
- Rep announcements  
- Real-time coaching feedback using gTTS + playsound  

Weekly Workout Plan Generator  
- User enters Age + Weight  
- Automatically generates 7-day push-up & sit-up plan  

---

## Demo Output

- Live webcam stream with pose skeleton overlay  
- Rep counter updates instantly  
- Feedback displayed + spoken aloud  
- Weekly plan shown on the dashboard  

---<img width="1024" height="1024" alt="Flowchart" src="https://github.com/user-attachments/assets/2dc2e11c-eeda-4193-a93d-383d6af857af" />




## Tech Stack

| Component | Technology |

|----------|------------|

| Pose Estimation | YOLOv8 Pose (Ultralytics) |

| Backend | Flask (Python) |

| Frontend | HTML + CSS + JavaScript |

| Audio Feedback | gTTS + playsound |

| Computer Vision | OpenCV |

| Math & Angles | NumPy |

---

## 📂 Project Structure

```bash
AI-Kinematics-Trainer/

│

├── app.py                     # Main Flask + Pose + Rep Counter App

├── templates/

│   └── index.html             # Dashboard UI

│

├── static/

│   ├── style.css              # Styling

│   └── pushup.png             # Exercise reference image

│

├── weights/

│   └── best.pt                # Custom YOLOv8 Pose Model

│

└── README.md
