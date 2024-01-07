import torch

import torch.nn.functional as F

X_3_4 = torch.rand(3,4)
W_4_5 = torch.rand(4,5)
b_1_5 = torch.rand(1,5)

H_3_5 = F.relu(torch.matmul(X_3_4, W_4_5) + b_1_5)

W_5_2 = torch.rand(5,2)
b_1_2 = torch.rand(1,2)

print(f"H_3_5:{H_3_5}\nW_5_2:{W_5_2}\nb_1_2:{b_1_2}")
O = torch.matmul(H_3_5, W_5_2) + b_1_2

print(O)


X, W_xh = torch.tensor(range(0, 3)).view(3, 1), torch.tensor(range(0, 4)).view(1, 4)
H, W_hh = torch.tensor(range(0, 12)).view(3, 4), torch.tensor(range(0, 16)).view(4, 4)

print(f"X:{X}\nW_xh:{W_xh}\nH:{H}\nW_hh:{W_hh}")

Y = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)


d1 = torch.cat((X, H), dim=1)
d0 = torch.cat((W_xh, W_hh), dim=0)
print(f"d1:{d1}\nd0:{d0}")

m = torch.matmul(d1, d0)
print(f"m:{m}")
print(f"Y:{Y}")