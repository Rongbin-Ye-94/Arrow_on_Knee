# Knee X-ray Classification with Clinical Priorities

This project implements a deep learning solution for classifying knee X-rays according to the Kellgren-Lawrence grading scale, with specific emphasis on clinical priorities. The model is designed to be particularly sensitive to severe cases while maintaining high precision for normal cases.

## Features

- ResNet-style CNN architecture with attention mechanisms
- Ordinal regression loss for better grade ordering
- Clinical priority-weighted evaluation metrics
- Comprehensive visualization tools
- Detailed logging and model checkpointing

## Clinical Priorities

The model is optimized for the following clinical priorities:
1. High recall for severe cases (KL4) - 40% weight
2. High precision for normal cases (KL0) - 30% weight
3. High recall for moderate cases (KL3) - 20% weight
4. High precision for moderate cases - 10% weight

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Structure

Organize your data in the following structure:
```
data/
└── Digital Knee X-ray Images/
    └── Knee X-ray Images/
        ├── MedicalExpert-I/
        │   ├── KL0/
        │   ├── KL1/
        │   ├── KL2/
        │   ├── KL3/
        │   └── KL4/
        └── MedicalExpert-II/
            ├── KL0/
            ├── KL1/
            ├── KL2/
            ├── KL3/
            └── KL4/
```

## Usage

1. Prepare your data as described above
2. Run the training script:
```bash
python Data_Camp.py
```

3. Monitor training progress:
- Check the `logs/` directory for training logs
- View saved models in `models/`
- Find evaluation results and visualizations in `results/`

## Model Architecture

The model uses:
- ResNet-style residual connections
- Attention mechanisms
- Batch normalization
- Dropout regularization
- Ordinal regression loss

## Evaluation Metrics

- Per-class precision and recall
- Cohen's Kappa with quadratic weights
- Confusion matrices
- Clinical utility score
- Mean Absolute Error

## Results

[Add your model's performance metrics and key results here]

## Contributing

[Add contribution guidelines if you want to accept contributions]

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Add acknowledgments for data sources, references, etc.]
- [Add any other credits or references]

## Contact

[Your Name] - [Your Email]

Project Link: [Your repository URL] 