import numpy as np
import pandas as pd
import pickle

with open("mnist_trained_model.pkl", "rb") as f:
    network = pickle.load(f)

import matplotlib.pyplot as plt
plt.subplot(121)
plt.plot(network.loss_history)
plt.subplot(122)
plt.plot(network.accuracy_history)
plt.show()