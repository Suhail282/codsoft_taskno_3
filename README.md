# codsoft_taskno_3
Artificial Intelligence task_no_3
Image Captioning AI

This project implements an **Image Captioning** system that automatically generates a descriptive caption for a given image. It combines **Convolutional Neural Networks (CNNs)** for image feature extraction and **RNN/Transformer models** for language generation.

What It Does

- Takes an image as input
- Extracts visual features using **ResNet50** (or VGG16)
- Uses an **LSTM model** (or Transformer) to generate a caption
- Outputs a human-readable sentence describing the image

Tech Stack

- **Python 3**
- **TensorFlow / Keras**
- **ResNet50 / VGG16** – Pretrained CNN for feature extraction
- **LSTM / GRU** – Caption generation
- **Tokenizer** – For converting text to sequences
- **Flickr8k / MSCOCO** dataset – Training data
  
How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/image-captioning-ai.git
   cd image-captioning-ai
2. Install dependencies: 
   pip install -r requirements.txt
3. Run preprocessing & training:
   python train.py
5. Caption an image:
   python caption.py --image test.jpg
