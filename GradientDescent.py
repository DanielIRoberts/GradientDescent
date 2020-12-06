# Daniel Roberts
# Daoqun Yang
# Advanced Gradient Descent

# Importing
import sys
import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Learner Class
class Learner:

    # Initialization
    def __init__(self, maxIter, alpha, beta, minChange):
      trainX, trainY, testX, testY = self.getData()
      weight, mse = self.trainModel(maxIter, alpha, beta, minChange, trainX, trainY)
      self.testModel(weight, testX, testY)

      # Prepping data for plotting
      self.mse = mse

    # Getting data and separating into training and test
    def getData(self):

      # Reading csv
      data = pd.read_csv("https://ydqexample1.s3.us-east-2.amazonaws.com/energydata_complete.csv")

      # Removing extra columns
      data = data.drop(columns = ["date", "rv1", "rv2", "lights"])

      # Scaling data
      scaler = StandardScaler()
      cols = range(1, len(data.columns) - 1)
      data.iloc[:, cols] = scaler.fit_transform(data.iloc[:, cols])

      # Adding intercept column
      data.insert(0, "Int", [1] * len(data), True)

      # Splitting x and y
      x = data.drop(columns = ["Appliances"])
      y = data.iloc[:, 1]

      # Separting data
      trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

      # Resetting indicies
      trainX.reset_index(drop = True, inplace = True)
      trainY.reset_index(drop = True, inplace = True)
      testX.reset_index(drop = True, inplace = True)
      testY.reset_index(drop = True, inplace = True)

      return trainX, trainY, testX, testY

    def trainModel(self, maxIter, alpha, beta, minChange, trainX, trainY):

      # Getting column and row length
      lenRow = len(trainX)
      lenCol = len(trainX.columns)

      # Initializing variables
      v = [0] * (lenCol)
      allMse = []
      mse = 0

      # Randomizing weight
      initial = random.randint(50, size = 1)
      weight = random.normal(-initial, initial, lenCol)

      for k in range(maxIter):
          # Getting h matrix
          h = trainX.dot(weight)

          # Getting error
          e = np.subtract(h, trainY)

          # Calculating MSE
          newMse = 0.5 * (1 / lenRow) * sum(e ** 2)

          # Calculating dMSE/dw_i
          dmse = [0] * (lenCol)

          for i in range(lenCol):
              dmse[i] = 0
              for j in range(lenRow):
                  dmse[i] = (e[j] * trainX.iloc[j, i] + dmse[i]) / lenRow

          # Calculating new weights with RMSprop
          for i in range(lenCol):
              v[i] = v[i] * beta +  (1 - beta) * (dmse[i] ** 2)
              weight[i] = weight[i] - (alpha * dmse[i]) \
                          / (v[i] ** (1/2) + 10 ** (-8))

          # Adding new mse to list
          allMse.append(newMse)

          # Checking to see if change was small
          if (abs(newMse - mse) <= minChange):
            mse = newMse
            break
          else:
            mse = newMse

      # Printing final mse
      print("\nThe final training MSE is: %f" % mse)

      return weight, allMse

    def testModel(self, weight, testX, testY):
        # Getting h matrix
        h = testX.dot(weight)

        # Getting error
        e = np.subtract(h, testY)

        # Calculating MSE
        mse = 0.5 * (1 / len(testX)) * sum(e ** 2)

        print("The test MSE is: %f\n" % mse)
        print("Error Values:")
        print(e)

if __name__ == "__main__":
  # Getting input
  maxIter = int(input("Please enter max iterations:"))
  alpha = float(input("Please enter desired alpha:"))
  beta = float(input("Please enter desired beta:"))
  plotNam = input("Enter name of the plot:")

  # Running learner
  learn = Learner(maxIter, alpha, beta, 0.1)

  # Plotting 
  plot = plt.plot(range(1, len(learn.mse) + 1), learn.mse)
  title = "MSE vs Iteration: " + "a = " + str(alpha) +  ", b = " + str(beta)
  plt.title(title)
  plt.xlabel("Iteration")
  plt.ylabel("Training MSE")
  plt.savefig(plotNam)