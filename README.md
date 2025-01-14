AI-Emotion-Based-Midi-Helper

in order to use, outside of src,
 make a folder(output/models/continuous_concat),
  and place the contents of https://drive.google.com/file/d/17Exfxjtq7bI9EKtEZlOrBCkx8RBx7h77/view?usp=sharing in it.

  then, to run, use streamlit run src/run_midi_generation/run_midi_generation_gui.py

  **Here's the updated README:

---

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

