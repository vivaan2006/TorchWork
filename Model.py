import torch
import torch.nn as nn
import torch.optim as optim

# normal model stuff, made for a linear model "nn.Linear()"

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.fc(x)

# Load the data that consists of 2 random 100 by 10 tensors

inputs = torch.randn(100, 10)
labels = torch.randn(100, 10)


# call the model class
# use mseLoss to calculate the error of the mean
# using optim to optimize the program and use stochiastic gradient descent, with the model parameters as an arg
# Define the loss function and the optimizer

model = Model()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model using a for loop, as long as epoch is ran 100 times, it does the forward and backward pass
for epoch in range(100):
  
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # Backward pass 
    optimizer.zero_grad()
    loss.backward()
    
    # Optimization
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print ('Epoch [{}/100], Loss: {:.4f}'.format(epoch+1, loss.item()))
