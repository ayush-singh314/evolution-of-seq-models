# Evolution of Sequence Models

This repository contains implementations and experiments exploring the evolution of sequence models in deep learning, from RNNs to modern Large Language Models.

## Project Structure

```
evolution-of-sequence-models/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── configs/                  # Configuration files for each model
│   ├── rnn.yaml             # RNN Encoder-Decoder config
│   ├── attention.yaml       # Attention-based RNN config
│   ├── transformer.yaml     # Transformer config
│   └── llm.yaml             # Large Language Model config
├── data/                     # Data directory
│   ├── raw/                 # Raw datasets
│   └── processed/           # Processed datasets
├── notebooks/               # Jupyter notebooks for analysis
│   ├── exploratory.ipynb    # Exploratory data analysis
│   └── analysis.ipynb       # Model analysis and results
├── src/                      # Source code
│   ├── dataset.py           # Dataset utilities
│   ├── tokenizer.py         # Tokenizer implementation
│   ├── train.py             # Training utilities
│   ├── evaluate.py          # Evaluation utilities
│   ├── utils.py             # General utilities
│   ├── models/              # Model implementations
│   │   ├── rnn_encoder_decoder.py    # RNN Encoder-Decoder
│   │   ├── attention.py     # Attention mechanisms
│   │   ├── transformer.py   # Transformer model
│   │   └── decoder_only_llm.py       # Decoder-only LLM
│   └── experiments/         # Experiment scripts
│       ├── experiment_rnn.py         # RNN experiment
│       ├── experiment_attention.py   # Attention RNN experiment
│       ├── experiment_transformer.py # Transformer experiment
│       └── experiment_llm.py         # LLM experiment
├── outputs/                  # Training outputs
│   ├── checkpoints/         # Model checkpoints
│   ├── logs/                # Training logs
│   └── plots/               # Result plots
└── report/                   # Final report
    └── final_report.pdf     # Comprehensive analysis report
```

## Models Implemented

### 1. RNN Encoder-Decoder (`rnn_encoder_decoder.py`)
- Basic sequence-to-sequence model using LSTM
- Configurable number of layers and bidirectionality
- Teacher forcing during training

### 2. Attention-based RNN (`attention.py`)
- Bahdanau (additive) attention
- Luong (multiplicative) attention
- Multi-head attention mechanisms
- Integration with RNN encoder-decoder

### 3. Transformer (`transformer.py`)
- Full encoder-decoder architecture
- Scaled dot-product attention
- Positional encoding
- Configurable number of layers and heads

### 4. Decoder-only LLM (`decoder_only_llm.py`)
- GPT-style architecture
- Causal self-attention
- Various generation strategies (greedy, top-k, top-p)
- Language modeling objective

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd evolution-of-seq-models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

Each model has its own experiment script:

```bash
# RNN Encoder-Decoder
python src/experiments/experiment_rnn.py

# Attention-based RNN
python src/experiments/experiment_attention.py

# Transformer
python src/experiments/experiment_transformer.py

# Large Language Model
python src/experiments/experiment_llm.py
```

### Configuration

Models can be configured using YAML files in the `configs/` directory. Each configuration includes:

- **Model architecture** (hidden size, number of layers, etc.)
- **Training parameters** (learning rate, batch size, epochs)
- **Data settings** (sequence length, vocabulary size)

### Custom Data

To use your own data:

1. Place raw data in `data/raw/`
2. Process and save to `data/processed/`
3. Modify the data loading functions in `dataset.py`

## Features

- **Modular Design**: Each component is designed to be reusable
- **Comprehensive Evaluation**: Multiple metrics including perplexity, accuracy, F1-score
- **Text Generation**: Support for various generation strategies
- **Logging and Checkpointing**: Automatic saving of model states and training logs
- **Reproducibility**: Seed setting and deterministic behavior

## Dependencies

See `requirements.txt` for the full list of dependencies. Key packages include:

- PyTorch
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Jupyter
- PyYAML

## Results

After running experiments, results will be saved in:

- `outputs/checkpoints/`: Model checkpoints
- `outputs/logs/`: Training logs
- `outputs/plots/`: Visualization plots
- Individual JSON files with metrics and training history

## Analysis

Use the Jupyter notebooks in `notebooks/` for:

- Exploratory data analysis
- Model performance comparison
- Visualization of training curves
- Error analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original papers and implementations that inspired this work
- Open-source community for various tools and libraries
- Research community for advancing sequence modeling

## Contact

For questions or suggestions, please open an issue in the repository.


“The LSTM baseline converged faster and achieved slightly lower perplexity in this small-scale character modeling task. The additive attention increased computational cost without significant gains, motivating exploration of fully attention-based Transformers.”