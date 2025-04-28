---

# Gym Exercise Processor Scripts

This project provides several command-line tools for gym exercise analysis using machine learning models. Below are the details of each script and how to use them.

---

## **1. Classifier Pipeline**

### **Description**
Runs the default classifier pipeline for gym exercise analysis.

### **Usage**
```bash
classifier_pipeline
```

### **Details**
- This script initializes the `DefaultClassifier` and runs its pipeline.
- No additional arguments are required.

---

## **2. Process Webcam Feed**

### **Description**
Processes a live webcam feed for gym exercise analysis.

### **Usage**
```bash
process_camera --checkpoint <path_to_checkpoint> --output <output_video_path> --seq_len <sequence_length> --fps <fps>
```

### **Arguments**
- `--checkpoint` (optional): Path to the model checkpoint file. Default: checkpoint.pth.
- `--output` (optional): Path to save the processed output video. Default: output.mp4.
- `--seq_len` (optional): Sequence length for the model. Default: `128`.
- `--fps` (optional): Frames per second for the output video. Default: `30`.

### **Example**
```bash
process_camera --checkpoint model_checkpoint.pth --output webcam_output.mp4 --seq_len 64 --fps 25
```

---

## **3. Process Video File**

### **Description**
Processes a video file for gym exercise analysis.

### **Usage**
```bash
process_video --checkpoint <path_to_checkpoint> --input <input_video_path> --output <output_video_path> --seq_len <sequence_length>
```

### **Arguments**
- `--checkpoint` (optional): Path to the model checkpoint file. Default: checkpoint.pth.
- `--input` (optional): Path to the input video file. Default: `input.mp4`.
- `--output` (optional): Path to save the processed output video. Default: output.mp4.
- `--seq_len` (optional): Sequence length for the model. Default: `128`.

### **Example**
```bash
process_video --checkpoint model_checkpoint.pth --input input_video.mp4 --output processed_output.mp4 --seq_len 128
```

---

## **Installation**

To install the project and make the scripts available as command-line tools, use the following command:

```bash
pip install -e .
```

This will register the following commands:
- `classifier_pipeline`
- `process_camera`
- `process_video`

---

## **Notes**
- Ensure that the required dependencies are installed. You can install them using the requirements.txt file:
  ```bash
  pip install -r requirements.txt
  ```
- Replace `<path_to_checkpoint>` and other placeholders with actual file paths or values as needed. There is a checkpoint is in repository.

---