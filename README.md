# Handwritten Digit Recognition using Neural Networks in PyTorch.

## To run the scripts:
### 1. Clone the repository:
```bash
  git clone https://github.com/AlexDuna/Handwritten_Digit_Recognition_NN.git
  cd Handwritten_Digit_Recognition_NN
```
### 2. Install dependencies:
You can install the required libraries with:
```
  pip install torch torchvision pillow
```

### 3. Train the model:
```
  python train_mnist_dataset.py
```

This will create a file:
```
  mnist_handwritten_digits.pth
```

### 4. Run the GUI:
```
  python draw_gui.py
```
 After that, you can draw, and get predictions based on your drawings :)

- The GUI should look like this:
- Default:
<img src="/Images/GUI.png" alt="isolated"/>

- Prediction 1:
<img src="/Images/Example1_GUI.png" alt="isolated"/>

- Prediction 2:
<img src="/Images/Example2_GUI.png" alt="isolated"/>



# How the algorithm actually works
<img src="/Images/Algorithm_Representation.png" alt="isolated"/>

## Handwritten Digit Recognition is a classification problem, where the goal is to correctly identify digits [0-9] from images.
- **Dataset: MNIST**
- Contains **70.000 grayscale images** of handwritten digits
  - 60.000 for training
  - 10.000 for testing
- Each image is **28 x 28** pixels
- Labels range from **0 to 9**

### Examples of MNIST dataset:
<img src="/Images/MNIST_dataset.png" alt="isolated"/>

## Step 1: Preprocess the Data
- before giving the images to a neural network, we need to **preprocess** them:

### Reshape and Normalize
- Convert the **28x28 pixels image** into a **1D array** of **784 values**
- In code, this is done inside the model using ```nn.Flatten()```
- Normalize the pixel values from **0 - 255** to **0 - 1** using ```ToTensor()```
  - ```ToTensor()``` converts a PIL image with ```[0, 255]``` to a tensor with values in ```[0, 1]``` 

**Before and after flattening:**
```Original (28x28)``` -> ```Flattened (1x784)```
<img src="/Images/Flatten.png" alt="isolated"/>

- So 784 pixels go as input of a Neural Network:
<img src="/Images/NN_input.png" alt="isolated"/>

## Step 2: Building the Neural Network
A **Neural Network** consists of:
1. **Input Layer** - 784 neurons of 28x28 pixels
2. **Hidden Layer** - to learn complex patterns
3. **Output Layer** - 10 neurons for digits 0-9

So the Neural Network looks like:
<img src="/Images/Layers.png" alt="isolated"/>
```Input (784)``` -> ```Hidden Layer``` -> ```Output (10)```

Detailing:
### Input Layer:
- Each pixel in the image corresponds to **one neuron**
- 784 neurons for **28x28 pixels**

### Hidden Layer:
Hidden Layers contain **neurons** that apply **weights1 and activation functions**.
- **Key Concepts:**
  - **Weights**: adjustable parameters that determine the importance of each pixel
  - **Bias**: added value to adjust activation
  - **Activation Function**: introduces non-linearity
- In code, I used:
  - ```nn.Linear``` for **weights** and **bias**
  - **ReLU** (**Rectified Linear Unit**) to introduce **non-linearity**
<img src="/Images/ReLU.png" alt="isolated"/>

- Formula:
<img src="/Images/hidden_layer_formula.png" alt="isolated"/>

- For a better understanding, here is an image example for a single neuron:
<img src="/Images/Single_Neuron.png" alt="isolated"/>

### Output Layer:
- Is the final layer that contains **10 neurons** (one for each digit 0-9)
- In the code, for this algorithm I used ```CrossEntropyLoss```, in PyTorch it internally applies **Softmax** to convert the raw outputs into probabilities in the range ```[0, 1]```
- This activation function converts outputs into **probabilities in range of 0-1**
<img src="/Images/Softmax.png" alt="isolated"/>

Example for Output Probabilities:
```
0 → 0.01
1 → 0.03
2 → 0.02
3 → 0.10
4 → 0.05
5 → 0.08
6 → 0.60 
7 → 0.80 (Highest Probability)
8 → 0.07
9 → 0.03
```

**Prediction** = 7 (highest probability)

## Step 3: Training the Neural Network
### Forward Propagation
```Input Layer -> Matrix Multiplication -> Activation -> Output```

### Loss Function (Cross-Entropy)
The difference between the predicted and actual output is calculated using **cross-entropy loss**:
<img src="/Images/Cross_Entropy.png" alt="isolated"/>

Where:
<img src="/Images/Cross_Entropy_Expl.png" alt="isolated"/>

### Backpropagation & Optimization
- **Backpropagation** updates weights using **gradient descent**
- **Optimization Algorithm: Adam**
  - **Gradient Descent:**
<img src="/Images/Gradient_Descent.png" alt="isolated"/>
Where:
<img src="/Images/Gradient_Descent_Expl.png" alt="isolated"/>

## Step 4: Testing & Prediction
- Trained model is tested on unseen data
- The accuracy is calculated
- The model predicts digits based on learned features

### Final Prediction Process:
```Input Layer -> Neural Network -> Prediction Output```

## Last step, was to create a GUI, where the user can draw a digit, and get a prediction
Pretty Cool :)
  

  

 
