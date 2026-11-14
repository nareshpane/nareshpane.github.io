import torch

w = torch.tensor(
    2.0, dtype=torch.float64,
    requires_grad=True
)

y = w**2 + 3*w + 2
y.backward()

print(float(w.grad))  # 7.0
