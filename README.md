# Image Captioning with TensorFlow

Generate captions for any image using the Flickr8k dataset with an attention-based neural network.

## Features

- Efficient data processing with tf.data.Dataset
- Transfer learning with InceptionV3 for image feature extraction
- Attention mechanism for focusing on relevant image parts
- Support for pre-trained word embeddings (GloVe)
- Comprehensive evaluation metrics including BLEU scores
- Visualization tools for attention weights

## Project Structure

```
ImageCaptioning/
├── preprocess_data.py      # Data preparation script
├── image_caption_generator.py  # Core implementation
├── train_caption_model.py  # Training script
├── generate_captions.py    # Inference script
├── Caption_Gen.ipynb       # Original notebook
└── README.md               # Project documentation
```

## Getting Started

### 1. Prepare the Data

Run the preprocessing script to prepare the captions and extract image features:

```bash
python preprocess_data.py
```

This will:
- Load and preprocess captions
- Split data into train/validation/test sets
- Create vocabulary and tokenize captions
- Extract image features using InceptionV3
- Load pre-trained word embeddings (if available)

### 2. Train the Model

Train the image captioning model:

```bash
python train_caption_model.py
```

This will:
- Create data generators
- Build and compile the attention-based caption model
- Train the model with early stopping
- Evaluate the model on test data
- Generate sample captions

### 3. Generate Captions

Generate captions for new images:

```bash
# Caption a single image
python generate_captions.py --image path/to/image.jpg

# Caption all images in a directory
python generate_captions.py --dir path/to/image/directory

# Specify a custom model
python generate_captions.py --image path/to/image.jpg --model path/to/model.h5
```

## Model Architecture

The model uses a hybrid CNN-RNN architecture:

1. **Image Encoder**: Pre-trained InceptionV3 to extract 2048-dimensional feature vectors from images
2. **Attention Mechanism**: Allows the model to focus on relevant parts of the image for each word
3. **Text Decoder**: LSTM network with embedding layer and attention to generate captions word by word

## Training and Evaluation

The model is trained using:
- Cross-entropy loss
- Adam optimizer with learning rate decay
- Early stopping to prevent overfitting

Evaluation metrics include:
- BLEU scores (BLEU-1 and BLEU-4)
- Accuracy and loss values

## Future Improvements

- Implement beam search for better caption generation
- Add support for more advanced attention mechanisms (e.g., multi-head attention)
- Experiment with Transformer-based architectures
- Fine-tune the CNN encoder for better feature extraction
- Add more evaluation metrics like METEOR and CIDEr
