# Alphanumeric Sign Language Translation System v4.0

A high-performance, intelligent desktop application designed to bridge communication gaps by translating hand sign gestures into text in real-time. Built with an optimized machine learning classification backbone and wrapped in a professional desktop interface.

---

## 🚀 System Architecture Features

* **Multi-Threaded UI Control:** Powered by PyQt6, the camera stream and core ML inference engines run inside an independent background worker thread, ensuring the hardware-accelerated user interface remains lag-free at a fluid frame rate.
* **Intelligent Hand Tracking:** Leverages Google MediaPipe's high-fidelity hand landmark pipelines to extract spatial tracking arrays.
* **Optimized Brain Model:** Evaluates and tracks gesture classification through machine learning structures, packaging the high-performing model seamlessly using `joblib`.
* **Advanced Text Modification Logic:**
  * `SPACE` - Injects word breaks smoothly.
  * `BACKSPACE` - Instantly strips the last character dynamically without polluting the string.
  * `CLEAR_WORD` - Strips text backward directly to the previous space index to drop entire words instantly.

---

## 🛠️ Installation & Setup

1. Clone the repository to your local workspace:
   ```bash
   git clone [https://github.com/prabal009/Sign-Language-Translator.git](https://github.com/prabal009/Sign-Language-Translator.git)
   cd Sign-Language-Translator