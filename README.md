# MNIST Handwritten Digit Classifier

A machine learning project that recognizes handwritten digits (0-9) using the famous **MNIST dataset** and an **SGD (Stochastic Gradient Descent) classifier**.

## What Does This Project Do?

This project trains a computer to recognize handwritten digits, just like how humans can read numbers written by hand. 

**For example:**
1. You write the number "5" on paper.
2. The computer looks at the image and says "That's a 5!".

It's trained on **70,000 examples** of handwritten digits from the MNIST dataset, which is like a huge collection of homework assignments with numbers written by thousands of different people.

### How Does It Work?

1. **Loads the Data**: Gets 70,000 images of handwritten digits (28×28 pixels each).
2. **Augments the Data**: Creates more training examples by slightly shifting each image up, down, left, and right (this helps the model learn better).
3. **Trains the Model**: Uses an SGD Classifier to learn patterns in the images.
4. **Tests Accuracy**: Checks how well it can recognize digits it hasn't seen before.
5. **Saves the Model**: Stores the trained model so you can use it later without retraining.

## Project Performance

- **Training Data**: 60,000 original images → 300,000 augmented images
- **Test Accuracy**: ~84.7% (correctly identifies about 85 out of 100 digits)
- **Training Time**: ~2-10 minutes on a modern CPU

---

## Installation & Setup

### 1. Prerequisites
- **Python 3.7** or higher
- **pip** (Python package installer)

### 2. Clone the Repository
```bash
git clone https://github.com/Azie88/SGD-Classifier.git
cd SGD-Classifier
```

### 3. Create a Virtual Environment
A virtual environment keeps this project's packages separate from your system Python.

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
*You'll know it's activated when you see `(.venv)` at the start of your command line.*

### 4. Install Required Packages
```bash
pip install -r requirements.txt
```

**This installs all the necessary libraries:**
- `numpy` - Number crunching
- `pandas` - Data handling
- `scikit-learn` - Machine learning tools
- `scipy` - Scientific computing
- `matplotlib` - Data visualization
- `jupyter` - Notebook interface

---

## How to Use

### Running the Notebook

You can use the classic Jupyter interface or VS Code:

- **VS Code (Recommended):**
    1. Open the project folder.
    2. Open `shifted_images.ipynb`.
    3. Select the kernel **Python (.venv)** in the top-right corner.
    4. Run the cells sequentially.

- **Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

---

### What Happens When You Run It
The notebook will:
- Download the MNIST dataset (if not already cached).
- Show you example digits.
- Create augmented training data.
- Train the classifier (~5-10 minutes).
- Display accuracy metrics.
- Save the trained model as `sgd_clf_multiclass.pkl`.

### Using the Trained Model
After training, you can load and use the saved model:

```python
import pickle
import numpy as np

# Load the model
with open('sgd_clf_multiclass.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict a digit (image should be 784 pixels - 28x28 flattened)
# prediction = model.predict([your_image_data])
# print(f"Predicted digit: {prediction[0]}")
```

---

## Project Structure
```
.
├── shifted_images.ipynb      # Main notebook with all the code
├── sgd_clf_multiclass.pkl    # Trained model (generated after running)
├── requirements.txt          # List of required Python packages
└── README.md                 # This file
```

---

## Understanding the Output

### Confusion Matrix
The notebook generates a confusion matrix - a grid showing:
- **Rows**: What the actual digit was
- **Columns**: What the model predicted
- **Diagonal**: Correct predictions (darker = more correct)
- **Off-diagonal**: Mistakes (lighter = fewer mistakes)

### Accuracy Scores
- **Training Accuracy (~79%)**: How well it learned from training data.
- **Test Accuracy (~85%)**: How well it works on new, unseen digits.

---

## Key Concepts Explained

### What is Data Augmentation?
We create more training examples by shifting images slightly. This helps the model recognize digits even if they're not perfectly centered.

### What is an SGD Classifier?
SGD (Stochastic Gradient Descent) is a learning algorithm that:
- Looks at images one at a time (or in small batches).
- Learns patterns gradually.
- Adjusts itself to make better predictions.

### What is Cross-Validation?
We split data into chunks and test the model multiple times to ensure it works consistently, not just by luck.

---

## Next Steps

After running this project, you could:
- Try different classifiers (Random Forest, Neural Networks).
- Add more augmentation types (rotation, scaling).
- Build a web app where users can draw digits.
- Experiment with other datasets (letters, objects).

## Credits

- **Dataset**: MNIST (Modified National Institute of Standards and Technology)
- **Libraries**: scikit-learn, NumPy, pandas, matplotlib

## Contributions :handshake:

Open an issue, submit a pull request or contact me for any contributions.

## Author :writing_hand:

Andrew Obando

<a href="https://www.linkedin.com/in/andrewobando/"><img align="left" src="https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white" alt="Andrew Obando | LinkedIn"/></a>
<a href="https://medium.com/@obandoandrew8">
![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)
</a>

---

Feel free to star ⭐ this repository if you find it helpful!
