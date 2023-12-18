import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Hyper parameters
input_size = 1
output_size = 1
num_epochs = 1000
learning_rate = 0.001

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# Linear regression model
model = nn.Linear(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Convert numpy arrays to torch tensors
inputs = torch.from_numpy(x_train)
targets = torch.from_numpy(y_train)

# train the model
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if((epoch + 1) % 10 == 0):
        print(f"Epoch {epoch + 1}/{num_epochs}, loss:{loss.item():.4f}")


# Plot the graph
model.eval()  # 将模型设置为评估模式
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.suptitle("Linear regression figure")
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