# Gradient_Descent_IRL

Gradient Descent IRL utilizes Q_learning for predicting values used within the neural network training loop.

Neural Network (NN) outputs a prediction for the state penalizing matrix, Q, and the control penalizing matrix, R.
Those values are the parameters for the reward function used in Q-learning, thus they are the inputs to the Q-learning function.

The Q-learning fucntion uses Q and R to train a linear in parameter NN (LIPNN) to control a system through dynamics or samples. 

# Usage

The Q-Learning file works by pressing "run", there is no other files needed for the NN to learn. This file needs to be used first as it saves an optimal policy used in the GD_IRL file.

The GD_IRL file is a jupyter notebook which would need to have every section run individually.
