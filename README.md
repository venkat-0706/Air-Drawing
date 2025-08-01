

# 🎨 Hand Gesture Drawing App

> Control your canvas using just your hand.
> **Draw, erase, and switch colors** in real-time using intuitive hand gestures detected via webcam.

---

![Python](https://img.shields.io/badge/Python-3.7+-blue?logo=python\&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-red?logo=google)
![UI](https://img.shields.io/badge/UI-Realtime--Overlay-orange)



---

## 🧠 Key Features

| Feature                       | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| 🎨 **Air Drawing**            | Draw lines on a canvas using your index finger in real-time                 |
| 🧼 **Erasing**                | Make a fist gesture to erase parts of the drawing                          |
| 🌈 **Color Palette Switching**| Hover over virtual palette to switch between red, green, and blue          |
| 🖥️ **Live Feedback UI**       | See gestures drawn live on the video stream with MediaPipe hand overlays   |
| ⚡ **Smooth Drawing Engine**  | High-performance canvas rendering with OpenCV and NumPy                    |

---

## ✋ Gesture Recognition Logic

| Gesture                          | Action            |
|----------------------------------|-------------------|
| ✍️ Index finger up only          | Draw              |
| ✊ All fingers down (fist)       | Erase             |
| ✌️ Two fingers up               | Color selection   |
| 🖐️ Open palm                    | Idle / reset      |

MediaPipe detects **21 landmarks per hand**. Gesture logic is implemented by comparing fingertip and lower-joint positions to determine finger states.

---

## 🔧 Tech Stack

### 🧪 Core Technologies

| Tech           | Role                                                   |
|----------------|--------------------------------------------------------|
| **Python 3.7+**| Programming Language                                   |
| **OpenCV**     | Webcam streaming, canvas rendering, drawing functions |
| **MediaPipe**  | Real-time hand tracking and 3D landmark detection     |
| **NumPy**      | Array operations for efficient image processing       |

### 🎨 Drawing System

| Component         | Tool/Function              |
|------------------|----------------------------|
| Line Drawing      | `cv2.line()`               |
| Circle Erasing    | `cv2.circle()`             |
| Canvas Overlay    | NumPy + OpenCV masking     |
| Color Palette UI  | On-screen rectangles       |

---

## 📥 Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/hand-gesture-drawing.git
cd hand-gesture-drawing

# Install required libraries
pip install opencv-python mediapipe numpy

# Run the application
python app.py
````

---

## 🖼️ Screenshots

| Drawing Mode              | Eraser Mode                | Color Palette              |
| ------------------------- | -------------------------- | -------------------------- |
| ![](screenshots/draw.png) | ![](screenshots/erase.png) | ![](screenshots/color.png) |

---

## 🧩 How It Works

1. 🖥️ OpenCV captures real-time video from the webcam
2. ✋ MediaPipe detects 21 hand landmarks
3. 🧠 Gesture logic determines if fingers are up/down
4. 🎨 Drawing or erasing actions are applied on a NumPy-based canvas
5. 🔁 The canvas is overlaid on the webcam stream for a real-time visual experience

---

## ⚙️ Customization Options

* 🎯 **Add New Colors:** Modify `colors[]` and `color_names[]` lists
* ✏️ **Change Brush/Eraser Size:** Update `brush_thickness` and `eraser_thickness`
* 📤 **Save Canvas:** Use `cv2.imwrite('drawing.png', canvas)`
* 👇 **Add More Gestures:** Expand the `fingers_up()` logic for new controls

---

## 📅 Future Improvements

* [ ] Gesture to clear entire canvas
* [ ] Option to save drawings via GUI button or gesture
* [ ] Streamlit/Flask web interface
* [ ] Add more brushes (dotted, calligraphy)

---

---

### ✅ Why This Works Better

* Pure markdown = more reliable across GitHub, GitLab, Bitbucket, etc.
* Respects dark/light modes.
* Easier to edit and maintain.

---

## 🙋 Author

Made with ❤️ by [Chandu](https://github.com/venkat-0706)


🔗 Connect on [Linkedin](https://www.linkedin.com/in/chandu-0706)

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to use and modify it as you wish!

---

<p align="center">
  If you like this project, consider giving it a ⭐ and sharing it!
</p>
```

---

### ✅ Next Steps for You:

* Replace placeholders like `yourusername`, image paths, and LinkedIn links.
* Add a demo GIF in `screenshots/demo.gif`
* Create high-quality screenshots of each mode (drawing, erasing, color switching).

