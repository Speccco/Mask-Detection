# Mask Detection App

This **Mask Detection** app uses a **YOLOv11** model to detect whether people in images are wearing a mask. The app takes an image as input, runs the mask detection model on it, and displays the results.

## Demo
[Try The Demo Here!](https://huggingface.co/spaces/Speccco/Mask-Detection)

## How to Use

1. **Upload an Image**:
   - Click the "Upload an image" button to select a photo containing people (JPG, PNG, or JPEG).
   
2. **Detection Process**:
   - The app will process the image and use the YOLOv11 model to detect faces and check whether a mask is being worn.
   
3. **View Results**:
   - The app will show the original image with bounding boxes around detected faces and indicate whether the individual is wearing a mask.

## Model

The app is powered by a **YOLOv11 model** trained to detect **masks** on faces.

### Trained Model:

- The mask detection model has been trained using a custom dataset containing face images with and without masks.

## Tech Stack

- **YOLOv11** for mask detection
- **Streamlit** for the interactive web app
- **PyTorch** for the deep learning framework
- **OpenCV** for image processing and visualization
- **Pillow** for image manipulation

## Installation

1. Clone this repository:
   ```bash
   git clone https://huggingface.co/spaces/your-username/mask-detection
   cd mask-detection
