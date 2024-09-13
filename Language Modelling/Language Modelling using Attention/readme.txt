HW4P2 File:- 
Part 1

Components
1. CNN_LSTM_Encoder
Convolutional Layers: Processes the input data through sequential convolutional layers to extract features. Each convolution is followed by batch normalization and ReLU activation for non-linearity and stability.
Bidirectional LSTM: Further processes the output of the convolutional layers. Using a bidirectional LSTM allows the model to capture contextual information from both past and future data points effectively.
pBLSTM Modules: Further pyramid BLSTM layers to reduce the sequence length and increase the depth of representation.
Permutation Operations: Used to permute dimensions for compatibility between different layers and operations.
2. Decoder
DecoderLayer: Consists of multiple sub-layers including two multi-head attention mechanisms (one self-attention and one encoder-decoder attention) and one position-wise feedforward network. Each sub-layer includes a residual connection followed by layer normalization.
Embedding and Positional Encoding: Converts target tokens into vectors and adds positional encodings to retain sequential information.
Final Linear and Softmax Layer: Projects the decoder output to the target vocabulary size and applies a softmax to predict the probability distribution of the next token.
Model Architecture
SpeechTransformer
Encoder: Processes the input audio features through the CNN_LSTM_Encoder, followed by a linear projection to match the dimension required by the transformer (d_model).
Decoder: Handles the autoregressive prediction of text from the encoded audio features using stacked DecoderLayers. It utilizes masked self-attention to prevent future information leakage and encoder-decoder attention to focus on relevant parts of the audio input.
Output: The final output is a sequence of tokens representing the text transcription of the input audio.

Part 2:- 

Components
1. EncoderLayer
Multi-Head Attention (MHA): Utilizes multiple heads to simultaneously process the input sequence, capturing different features from different subspaces, followed by dropout.
FeedForward Network (FFN): A linear transformation applied after MHA to introduce non-linearity, followed by dropout.
Layer Normalization: Normalizes the output of each sub-layer (MHA and FFN), stabilizing the learning process.
Residual Connections: Facilitates deeper architectures by allowing gradients to flow through layers directly.
2. Encoder
CNN-LSTM Encoder: Extracts features from the input data, combining convolutional layers for local feature extraction and LSTM for sequence modeling.
Projection: Transforms the output from the CNN-LSTM Encoder to match the model dimension (d_model).
Positional Encoding: Adds information about the sequence order to the model, compensating for the lack of inherent sequence processing capability in the Transformer architecture.
Encoder Layers: A stack of EncoderLayer instances, allowing the model to learn complex patterns and interactions in the data.
3. Decoder
DecoderLayer: Similar to the EncoderLayer but includes cross-attention to focus on relevant parts of the input sequence based on the encoder's output.
Embedding and Positional Encoding: For the target sequence, converting tokens to vectors and adding positional information.
Final Linear and Softmax Layer: Transforms the decoder output to the vocabulary size for prediction, followed by a softmax for probability distribution over possible tokens.
Model Architecture
FullTransformer
Encoder: Processes the input data through a CNN-LSTM Encoder, projection layer, positional encoding, and multiple EncoderLayers. It outputs a continuous representation of the input.
Decoder: Takes the encoder output and the target sequence, processes it through several DecoderLayers with attention mechanisms focusing on different parts of the encoder output, and predicts the next token in the sequence.
Greedy Search Decoding: In the recognize method, the decoder performs a greedy search to generate output sequences one token at a time based on maximum probability.
