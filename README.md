# CNN-Transformer-Image-Captioning   README WIP

This project implements an image captioning model that integrates computer vision and natural language processing using a CNN–Transformer architecture.  
It automatically generates natural-language captions for input images, demonstrating how visual and linguistic representations can be combined within a single deep learning system.

---

## Objective
To train a model that can recognise visual content and describe it in plain English, transforming images into coherent textual captions.

**Example**

Input: an image of a dog running on grass  
Output: "a dog running through a field"

---

## Architecture Overview
- **Encoder:** EfficientNet-B0 (pretrained CNN) extracts visual features.  
- **Decoder:** Transformer-based language model generates textual descriptions.  
- **Tokenizer:** Basic English tokenizer from TorchText with a custom vocabulary.  
- **Dataset:** [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) - contains 8k images, each with five captions.

The encoder is frozen during training for efficiency, while the Transformer learns to map feature embeddings to grammatically consistent captions.

---

## How to Run

The notebook is designed to run in **Google Colab** with the Flickr8k dataset and the model weights stored in your Google Drive.

### 1. Download the dataset
Download the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) from Kaggle.  
After extracting it, you should have a folder that includes:

/Images/ # Folder containing all Flickr8k images

captions.txt # Text file containing image:caption pairs

Upload this folder to your Google Drive.  

### 2. Download pretrained weights
Download the pretrained model weights file (`image_caption_weights.pt`) provided in this repository and place it directly in your Google Drive.
