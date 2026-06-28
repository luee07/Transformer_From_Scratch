# Transformer From Scratch in PyTorch

A complete implementation of the **Transformer architecture** from scratch using **PyTorch**, built to understand every core component introduced in the paper **"Attention Is All You Need"**.

This project implements the Transformer **without relying on `torch.nn.Transformer`**, covering the encoder-decoder architecture, multi-head attention, positional encoding, masking, training pipeline, and inference.

---

## 📖 Overview

The Transformer architecture revolutionized Natural Language Processing by replacing recurrent networks with self-attention mechanisms, enabling parallel computation and significantly improving performance on sequence-to-sequence tasks.

This project recreates the Transformer step-by-step to provide a deeper understanding of its mathematical foundations and implementation details.

---

## ✨ Features

- Complete Transformer implementation from scratch
- Token Embeddings
- Sinusoidal Positional Encoding
- Multi-Head Self Attention
- Multi-Head Cross Attention
- Position-wise Feed Forward Network
- Residual Connections
- Layer Normalization
- Encoder Stack
- Decoder Stack
- Padding Mask
- Look-Ahead (Causal) Mask
- Sequence-to-Sequence Architecture
- Training Pipeline
- Translation Inference
- Checkpoint Saving & Loading

---

## 🏗️ Project Structure

```text
Transformer_From_Scratch/
│
├── model/
│   ├── embeddings.py
│   ├── positional_encoding.py
│   ├── multi_head_attention.py
│   ├── multi_head_cross_attention.py
│   ├── feed_forward.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── transformer_encoder.py
│   ├── transformer.py
│   └── mask.py
│
├── data/
│   ├── tokenizer.py
│   ├── vocabulary.py
│   ├── translation_dataset.py
│   └── collate.py
│
├── config.py
├── train.py
├── translate.py
├── evaluate.py
├── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 Transformer Architecture

```
                 Input Tokens
                       │
               Token Embedding
                       │
          Positional Encoding
                       │
              Encoder Stack × N
                       │
               Encoder Output
                       │
────────────────────────────────────────
                       │
              Target Embedding
                       │
          Positional Encoding
                       │
        Masked Multi-Head Attention
                       │
          Cross Multi-Head Attention
                       │
         Position-wise Feed Forward
                       │
              Decoder Stack × N
                       │
                Linear Layer
                       │
                 Softmax Output
```

---

## 🛠️ Technologies Used

- Python
- PyTorch
- TorchText (optional)
- NumPy
- tqdm

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Transformer_From_Scratch.git
```

Move into the project directory:

```bash
cd Transformer_From_Scratch
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

Train the Transformer:

```bash
python train.py
```

The training script:

- Loads the dataset
- Builds vocabularies
- Creates DataLoader
- Trains the Transformer
- Computes training loss
- Saves checkpoints

---

## 🔍 Inference

Translate a sentence:

```bash
python translate.py
```

---

## 📊 Evaluation

Evaluate the trained model:

```bash
python evaluate.py
```

---

## 📚 Concepts Implemented

- Self Attention
- Scaled Dot Product Attention
- Multi-Head Attention
- Cross Attention
- Positional Encoding
- Residual Connections
- Layer Normalization
- Feed Forward Networks
- Encoder
- Decoder
- Teacher Forcing
- Sequence Masking
- Padding Mask
- Look-Ahead Mask
- Greedy Decoding

---

## 📈 Future Improvements

- Train on Multi30k Dataset
- BLEU Score Evaluation
- Beam Search Decoding
- Mixed Precision Training
- Learning Rate Warmup
- Label Smoothing Improvements
- TensorBoard Logging
- Attention Visualization
- Distributed Training
- Hugging Face Dataset Support

---

## 📖 References

- Vaswani et al. (2017), *Attention Is All You Need*
- The Annotated Transformer
- PyTorch Documentation

---

## 🎯 Learning Outcomes

Through this project, I gained hands-on experience with:

- Building Transformers from scratch
- Understanding self-attention mathematically
- Encoder-decoder architectures
- Sequence-to-sequence learning
- Training deep learning models in PyTorch
- Efficient masking strategies
- Neural machine translation fundamentals

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

Feel free to fork the repository and open a Pull Request.

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates further open-source contributions.
