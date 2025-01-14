# AI-MIDI-Helper

## Setup Instructions

### 1. Folder Structure
Create the following directory structure in the project root:
```
models/continuous_concat/
```

### 2. Download Model Files
Download the required model files from [this link](https://drive.google.com/file/d/17Exfxjtq7bI9EKtEZlOrBCkx8RBx7h77/view?usp=sharing).  
Extract the contents into the `models/continuous_concat/` directory.

### 3. Run the Application
From the project root directory, use the following command to start the application:
```bash
python3 -m src.midi_app.app
```

---

## About the Project

This application was built using [midi-emotion](https://github.com/serkansulun/midi-emotion) as a base. Specifically, the following components were adapted from the original repository:
- The **model** (stayed the same)
- The **generate.py** script (changed)
- The **data/** directory structure (stayed the same)

---
