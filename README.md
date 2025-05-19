# Knee X-ray Classification with Clinical Priorities

"I used to be a traveler like you until i took an arrow to the knee" -> Quote from The Elder Scrolls. 

An arrow in the knee could be removed but chronic discomfort of knee osteoarthritis is painful and lasting.
Based on the data provided by GBD(2019), till 2019, about 528 million people were affected by this disease around the world and it is a 110% increase comparing to 1990. Furthermore, 73% of patients were senior citizens over 55 Years old and 60% were female. Patients will experience pain, swelling and stiffness, which like the joke in Elder Scrolls, largely deteorites the ability to travel freely.

An early detection and proper treatment could largely help people to relieve such pain. 

The major method to detect the severity and plan for treatment accordingly is to use the Kellgren and Lawrence (K&L) grading system, an ordinal system from 0 to 4 as the gold standard for assessing knee osteoarthritis severity. 

This project utilizes a deep learning solution for classifying knee X-rays, with specific emphasis on clinical priorities. Comparing to the traditional models focused on the accuracy, the model used (improved Net) is designed to be particularly sensitive to severe cases while maintaining high precision for normal cases, that are more influencial to the patients and avoiding misdiagonsis and delay of the treatment in the clinical. 

This is a project intends to help with the efforts to fight such disease. Although none of us were the Last Dragonborn, but we should be able to travel freely around continents for our own life adventures. 

References:
GBD, 2019; Global burden of 369 diseases and injuries in 204 countries and territories, 1990–2019: a systematic analysis for the Global Burden of Disease Study 2019. https://vizhub.healthdata.org/gbd-results/.

WHO, 2023; Osteoarthritis. Retrieved by Rongbin Ye on 05/28/2025 from: https://www.who.int/news-room/fact-sheets/detail/osteoarthritis#:~:text=With%20a%20prevalence%20of%20365,benefit%20from%20rehabilitation%20(3).

## Features

- ResNet-style CNN architecture with attention mechanisms
- Ordinal regression loss for better grade ordering
- Clinical priority-weighted evaluation metrics
- Comprehensive visualization tools
- Detailed logging and model checkpointing

## Clinical Priorities
Instead of having pure model based metrics, this code integrate the potential loss to the patients and 
business costs to the doctors. Therefore, optimize the selection criteria based on the clinial priorities accordingly. 

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

The Developers used two version of NN structures. From simple to complex, the general scores with defined scrore has been improved significantly. 

## Contributing
This is a practice project that using to integrate different evaluation. Will add further features and advanced model for diagnosis comparison. 

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- I am encouraged by DataCamp course to utilize the NN to further solve the problems such as Computer Vision, rather than traditional machine learning. 
- Kaggle Source：https://www.kaggle.com/datasets/orvile/digital-knee-x-ray-images


## Contact

Rongbin Ye - rongbin.ye.94@gmail.com
Jiaqi Chen - ronanchen0901@gmail.com

Project Link: https://github.com/Rongbin-Ye-94/Arrow_on_Knee
