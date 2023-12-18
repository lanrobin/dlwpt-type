import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")

from common_utils import inspect_tensors

# Hyper parameters
input_size = 1
output_size = 1
num_epochs = 10
learning_rate = 0.001

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
def backward_hook(module, grad_input, grad_output):
    print(f"--backward_hook--:{type(model)}")
    model.print_weight()

def pre_backward_hook(module, grad_output):
    print(f"--pre_backward_hook--:{type(model)}")
    model.print_weight()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # 输入和输出的维度都是1
        self.register_full_backward_pre_hook(pre_backward_hook)
        self.register_full_backward_hook(backward_hook)

    def forward(self, x):
        print("--forward--")
        out = self.linear(x)
        return out

    def zero_grad(self, set_to_none: bool = True) -> None:
        print("--zero_grad--")
        return super().zero_grad(set_to_none)
    
    def print_weight(self):
        print("--print_weight--")
        inspect_tensors([self.linear.weight])
    
    def print_parameters(self):
        print("--print_parameters--")
        object_attributes = dir(self.linear)
        # Print the result
        print("Attributes of the object:")
        for attribute in object_attributes:
            print(attribute)
        print("**print_parameters**")
    
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
model.print_parameters()

# Convert numpy arrays to torch tensors
inputs = torch.from_numpy(x_train)
targets = torch.from_numpy(y_train)

# train the model
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    #model.print_weight()
    print("--optimizer.zero_grad--")
    optimizer.zero_grad()
    #model.print_weight()
    print("--loss.backward--")
    loss.backward()
    #model.print_weight()
    print("--optimizer.step--")
    optimizer.step() # Here optimizer will update the model parameters.
    model.print_weight()
    print("*" * 50)

    if((epoch + 1) % 10 == 0):
        print(f"Epoch {epoch + 1}/{num_epochs}, loss:{loss.item():.4f}")

# Plot the graph
model.eval()  # 将模型设置为评估模式
predicted = model(torch.from_numpy(x_train)).detach().numpy()

plt.title("Linear regression")
plt.xticks(ticks=[0,2,4,6,8,10,12])
plt.yticks(ticks=[0,1,2,3,4])
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
fig = plt.gcf()
# Set the window title
fig.canvas.manager.set_window_title('Linear regression figure')
plt.legend()
plt.show()