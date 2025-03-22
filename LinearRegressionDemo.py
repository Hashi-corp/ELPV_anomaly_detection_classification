#https://www.geeksforgeeks.org/ml-linear-regression/ : required LR Evaluation metrics: must read
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation

#C:\A_HASH\College\Code\Python\AI_ML_Internship\data_for_lr.csv
url = 'C:\A_HASH\College\Code\Python\AI_ML_Internship\data_for_lr.csv'
data = pd.read_csv(url)
data

# Drop the missing values
data = data.dropna()

# training dataset and labels
train_input = np.array(data.x[0:500]).reshape(500, 1)
train_output = np.array(data.y[0:500]).reshape(500, 1)

# valid dataset and labels
test_input = np.array(data.x[500:700]).reshape(199, 1)
test_output = np.array(data.y[500:700]).reshape(199, 1)

class LinearRegression: 
    def __init__(self): 
        self.parameters = {} 

    def forward_propagation(self, train_input): 
        m = self.parameters['m'] 
        c = self.parameters['c'] 
        predictions = np.multiply(m, train_input) + c 
        return predictions 

    def cost_function(self, predictions, train_output): 
        cost = np.mean((train_output - predictions) ** 2) 
        return cost 

    def backward_propagation(self, train_input, train_output, predictions): 
        derivatives = {} 
        df = (predictions-train_output) 
        # dm= 2/n * mean of (predictions-actual) * input 
        dm = 2 * np.mean(np.multiply(train_input, df)) 
        # dc = 2/n * mean of (predictions-actual) 
        dc = 2 * np.mean(df) 
        derivatives['dm'] = dm 
        derivatives['dc'] = dc 
        return derivatives 

    def update_parameters(self, derivatives, learning_rate): 
        self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm'] 
        self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc'] 

    def train(self, train_input, train_output, learning_rate, iters): 
        # Initialize random parameters 
        self.parameters['m'] = np.random.uniform(0, 1) * -1
        self.parameters['c'] = np.random.uniform(0, 1) * -1

        # Initialize loss 
        self.loss = [] 

        # Initialize figure and axis for animation 
        fig, ax = plt.subplots() 
        x_vals = np.linspace(min(train_input), max(train_input), 100) 
        line, = ax.plot(x_vals, self.parameters['m'] * x_vals +
                        self.parameters['c'], color='red', label='Regression Line') 
        ax.scatter(train_input, train_output, marker='o', 
                color='green', label='Training Data') 

        # Set y-axis limits to exclude negative values 
        ax.set_ylim(0, max(train_output) + 1) 

        def update(frame): 
            # Forward propagation 
            predictions = self.forward_propagation(train_input) 

            # Cost function 
            cost = self.cost_function(predictions, train_output) 

            # Back propagation 
            derivatives = self.backward_propagation( 
                train_input, train_output, predictions) 

            # Update parameters 
            self.update_parameters(derivatives, learning_rate) 

            # Update the regression line 
            line.set_ydata(self.parameters['m'] 
                        * x_vals + self.parameters['c']) 

            # Append loss and print 
            self.loss.append(cost) 
            print("Iteration = {}, Loss = {}".format(frame + 1, cost)) 

            return line, 
        # Create animation 
        ani = FuncAnimation(fig, update, frames=iters, interval=200, blit=True) 

        # Save the animation as a video file (e.g., MP4) 
        ani.save('linear_regression_A.gif', writer='ffmpeg') 

        plt.xlabel('Input') 
        plt.ylabel('Output') 
        plt.title('Linear Regression') 
        plt.legend() 
        plt.show() 

        return self.parameters, self.loss 

linear_reg = LinearRegression()
parameters, loss = linear_reg.train(train_input, train_output, 0.0001, 20)
'''





''' My attempt #1 of creating an ML regression model '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Read data from CSV file (assuming 'test.csv' is in your working directory)
try:
    data = pd.read_csv("C:\A_HASH\College\Code\Python\AI_ML_Internship\\test.csv")
    data = pd.DataFrame(data)
except FileNotFoundError:
    print("Error: File 'test.csv' not found. Please check the file path.")
    exit()

# Optional: View scatterplot for x vs y data
# plt.scatter(data['x'], data['y'])
# plt.title("X vs Y scatterplot")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# Optional: Print correlations (uncomment if needed)
# print(data.corr)

# Handle missing values (consider different strategies if necessary)
print("Missing values before preprocessing:")
print(data.isnull().sum())

data.dropna(inplace=True)

print("Missing values after dropping rows with NaN:")
print(data.isnull().sum())

# Check data shapes before splitting (uncomment if needed)
# print(data['x'].shape)
# print(data['y'].shape)

# Split data into training and testing sets with consistent splitting
xtrain, xtest, ytrain, ytest = train_test_split(data['x'], data['y'], test_size=0.2)

# Reshape training and testing data to ensure consistent dimensions
xtrain = xtrain.values.reshape(-1, 1)
ytrain = ytrain.values.reshape(-1, 1)
xtest = xtest.values.reshape(-1, 1)
ytest = ytest.values.reshape(-1, 1)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Predict using the trained model
ypredict = model.predict(xtest)

# Visualize results (scatter plot with data and prediction)
plt.scatter(xtest, ytest, label='Actual')
plt.plot(xtest, ypredict, label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Linear Regression Results")
plt.legend()
plt.show()

# Evaluate model performance
mse = mean_squared_error(ytest, ypredict)
r2 = r2_score(ytest, ypredict)
print("Mean Squared Error (MSE):", mse)
print("R-squared:", r2)
