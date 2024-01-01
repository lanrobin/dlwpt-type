import torch

x = torch.ones(3,3, requires_grad=True)
print(x)
print(x.grad_fn)
print ("*" * 20)
y = x + 2

print(x)
print(x.grad_fn)

print(y)
print(y.grad_fn)
print ("*" * 20)
z = y/ 2

print(z)
print(z.grad_fn)
print ("*" * 20)

g = y * y * 3
out = g.mean()
print(g, out)
print ("*" * 20)

out.backward()
print(x.grad)
print ("*" * 20)