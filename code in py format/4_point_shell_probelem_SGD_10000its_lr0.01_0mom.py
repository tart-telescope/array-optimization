# Import packages 
import torch
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch import sqrt
import numpy as np

# create random initial coordinates
x1_0=np.random.normal(1)
y1_0=np.random.normal(1)
x2_0=np.random.normal(1)
y2_0=np.random.normal(1)
x3_0=np.random.normal(1)
y3_0=np.random.normal(1)
x4_0=np.random.normal(1)
y4_0=np.random.normal(1)

# Turn initial coordinates into tensors
x1=(torch.tensor(x1_0))
y1=(torch.tensor(y1_0))
x2=(torch.tensor(x2_0))
y2=(torch.tensor(y2_0))
x3=(torch.tensor(x3_0))
y3=(torch.tensor(y3_0))
x4=(torch.tensor(x4_0))
y4=(torch.tensor(y4_0))

# Apply grad=true
x1.requires_grad_(requires_grad=True) 
y1.requires_grad_(requires_grad=True) 
x2.requires_grad_(requires_grad=True) 
y2.requires_grad_(requires_grad=True) 
x3.requires_grad_(requires_grad=True) 
y3.requires_grad_(requires_grad=True) 
x4.requires_grad_(requires_grad=True) 
y4.requires_grad_(requires_grad=True)


# Plot random initial coordinates
print("Init Coords")
print(x1.detach().numpy(),y1.detach().numpy())
print(x3.detach().numpy(),y3.detach().numpy())
print(x2.detach().numpy(),y2.detach().numpy())
print(x4.detach().numpy(),y4.detach().numpy())

# Define the function area function
def compute_area(x1, y1, x2, y2, x3, y3, x4, y4):

    return 0.5*(torch.abs(x1*y2 -x1*y4 -x3*y2 + x3*y4 -y1*x2 +y1*x4 +y3*x2 -y3*x4))
  
# Define the function perimeter function
def compute_perimeter(x1, y1, x2, y2, x3, y3, x4, y4):
   
    s1 = torch.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    s2 = torch.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    s3 = torch.sqrt((x4 - x3)**2 + (y4 - y3)**2)
    s4 = torch.sqrt((x1 - x4)**2 + (y1 - y4)**2)
    return s1 + s2 + s3 + s4

# Define the optimizer and hyper-parameters
optimizer = optim.SGD([x1, y1, x2, y2, x3, y3, x4, y4], lr=.01, momentum=0)

# Number of iterations 
steps =10000

# perimeter constraint...limmited to 100 units
max_perimeter = 100

#collecting the value of the loss function for each interation 
loss_history = []

# Optimization loop
for step in range(steps):
    
    area = compute_area(x1, y1, x2, y2, x3, y3, x4, y4)
    
    perimeter = compute_perimeter(x1, y1, x2, y2, x3, y3, x4, y4)
    
    perimeter_penalty = (torch.max(torch.tensor(0.), ((perimeter)**2 - (max_perimeter)**2)**1)) #hmmmm
    
    # loss function
    loss = (torch.log(1/area) + perimeter_penalty**1)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    # Update parameters
    optimizer.step()

    loss_history.append(loss.item())

    if step % 100 == 0:
        print(f'Step {step}: loss_history = {loss.item()}') # Prints a value for loss function once every hundred iterations

# optomised coords
optimized_x1 = x1.item()
optimized_y1 = y1.item()
optimized_x2 = x2.item()
optimized_y2 = y2.item()
optimized_x3 = x3.item()
optimized_y3 = y3.item()
optimized_x4 = x4.item()
optimized_y4 = y4.item()

# compute optimized area 
dele=compute_area(x1, y1, x2, y2, x3, y3, x4, y4).item()

# Print the optimized coordinates, area, and perimeter..
print("Optimized Coordinates:")
print("Vertex 1: ({}, {})".format(optimized_x1, optimized_y1))
print("Vertex 2: ({}, {})".format(optimized_x2, optimized_y2))
print("Vertex 3: ({}, {})".format(optimized_x3, optimized_y3))
print("Vertex 4: ({}, {})".format(optimized_x4, optimized_y4))
print("Optimized Perimeter: {:.4f}".format(compute_perimeter(x1, y1, x2, y2, x3, y3, x4, y4).item()))
print("Optimized Area: {:.4f}".format(compute_area(x1, y1, x2, y2, x3, y3, x4, y4).item()))
print("Theoretical maximum = 625")

# Plot optimized coordinates
plt.plot(x1_0, y1_0, 'x', color='red', label="Initial Positions")
plt.plot(x2_0, y2_0, 'x', color='red')
plt.plot(x3_0, y3_0, 'x', color='red')
plt.plot(x4_0, y4_0, 'x', color='red')


plt.plot(x1.item(), y1.item(), 'o', color='black', label="Final Positions")
plt.plot(x2.item(), y2.item(), 'o', color='black')
plt.plot(x3.item(), y3.item(), 'o', color='black')
plt.plot(x4.item(), y4.item(),'o', color='black')
plt.title("Optimising Area of a Quadrlateral While Constraining the Perimeter")
plt.xlabel("Unit Length")
plt.ylabel("Unit Length")
plt.legend()
plt.grid(color='gray', linestyle='-', linewidth=0.5)
plt.xlim(-25, 25)
plt.ylim(-25, 25)
plt.savefig("4_points_init_final_positions_SGD_10000its_lr0.01_0mom.pdf")

# Plot loss history
plt.clf()
plt.plot(loss_history)
plt.title("Loss Function for Quadrilateral Area Optimization with Perimeter Constraint")
plt.xlabel("Iterations")
plt.ylabel("loss")
plt.grid(color='gray', linestyle='-', linewidth=0.5)
plt.savefig("4_points_loss_history_SGD_10000its_lr0.01_0mom.pdf")
