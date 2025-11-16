# UFC Fight Prediction Model: Makhachev vs Della Maddalena

A machine learning model that predicts the outcome of the UFC fight between Islam Makhachev and Jack Della Maddalena using real-world fighter statistics and historical fight data.

## Features

- **Real-world fighter data**: Uses actual UFC statistics including records, win rates, grappling/striking scores, and finishing rates
- **Machine learning model**: Trained on historical UFC fight patterns using Random Forest and Gradient Boosting classifiers
- **Comprehensive analysis**: Provides detailed fighter comparison, key advantages, and fight breakdown
- **Probability-based prediction**: Outputs win probabilities for both fighters

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python fight_prediction_model.py
```

## Model Details

The model analyzes:
- Win/loss records and win rates
- Grappling and striking abilities
- Submission and knockout rates
- Takedown accuracy
- Weight class experience
- Recent form and championship experience

## Fighter Profiles

**Islam Makhachev:**
- Record: 27-1 (96.4% win rate)
- Elite grappler with exceptional wrestling
- 52% submission rate
- Moving up from lightweight to welterweight

**Jack Della Maddalena:**
- Record: 18-2 (90% win rate)
- Elite striker with knockout power
- 67% knockout rate
- Current welterweight champion

## Prediction Methodology

The model uses an ensemble approach combining:
- Random Forest Classifier
- Gradient Boosting Classifier

Features are normalized and the model is trained on 500 simulated historical fights that follow realistic UFC patterns.

## Disclaimer

This is a predictive model based on statistical analysis. Actual fight outcomes depend on many unpredictable factors including injuries, strategy, and in-fight dynamics.

