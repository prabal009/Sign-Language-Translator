# Alphanumeric Sign Language Translation System v4.0

A high-performance, intelligent desktop application designed to bridge communication gaps by translating hand sign gestures into text in real-time. Built with an optimized machine learning classification backbone and wrapped in a professional desktop interface.

---

##  System Architecture Features

* **Multi-Threaded UI Control:** Powered by PyQt6, the camera stream and core ML inference engines run inside an independent background worker thread, ensuring the hardware-accelerated user interface remains lag-free at a fluid frame rate.
* **Intelligent Hand Tracking:** Leverages Google MediaPipe's high-fidelity hand landmark pipelines to extract spatial tracking arrays.
* **Optimized Brain Model:** Evaluates and tracks gesture classification through machine learning structures, packaging the high-performing model seamlessly using `joblib`.
* **Advanced Text Modification Logic:**
  * `SPACE` - Injects word breaks smoothly.
  * `BACKSPACE` - Instantly strips the last character dynamically without polluting the string.
  * `CLEAR_WORD` - Strips text backward directly to the previous space index to drop entire words instantly.
  * now how every letter is represented , you can take help of the images underneath 
<img width="740" height="740" alt="1" src="https://github.com/user-attachments/assets/b834f4a9-0003-4083-abf5-07d38aa5bd8e" />
<img width="424" height="509" alt="2" src="https://github.com/user-attachments/assets/06e05815-47ac-4c15-ac5f-d01710418ea9" />
<img width="600" height="460" alt="3" src="https://github.com/user-attachments/assets/8cba7724-aef0-4717-ab8b-5afeafebe38b" />
also all the letters are represented with single hand ( only j and s requires the use of both hands )
0 - both index fingers pointing upwards
1 - both pinky fingers pointing upwards
space - lay your palm facing towards ground
backspace for a letter - use single hand middle finger pointing upwards
backspace for a word - use both hand middle fingers pointing upwards




---

##  Installation & Setup

Copy and paste the unified script block below into your terminal to clone the project, isolate dependencies, install packages, and deploy the application. 

###  For Windows (CMD / PowerShell)
```bash
# 1. Clone the repository and navigate into the project directory
git clone [https://github.com/prabal009/Sign-Language-Translator.git](https://github.com/prabal009/Sign-Language-Translator.git)
cd Sign-Language-Translator

# 2. Create an isolated Python virtual environment to prevent dependency version conflicts
python -m venv venv

# 3. Activate the virtual environment
.\venv\Scripts\activate

# 4. Upgrade pip and install all computer vision, ML, and UI packages at once
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Launch the pre-trained real-time desktop interface directly
python app.py
```


 For macOS /  Linux
```

# 1. Clone the repository and navigate into the project directory
git clone [https://github.com/prabal009/Sign-Language-Translator.git](https://github.com/prabal009/Sign-Language-Translator.git)
cd Sign-Language-Translator

# 2. Create an isolated Python virtual environment to prevent dependency version conflicts
python3 -m venv venv

# 3. Activate the virtual environment
source venv/bin/activate

# 4. Upgrade pip and install all computer vision, ML, and UI packages at once
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Launch the pre-trained real-time desktop interface directly
python app.py
```

Custom Pipeline Execution
```

# Step A: Collect custom data frames (Webcam records your structural gesture frames)
python collect_data.py

# Step B: Train the model (Evaluates structures, exports optimized engine to joblib file)
python train_model.py

# Step C: Boot up the application utilizing your custom-trained dataset parameters
python app.py





