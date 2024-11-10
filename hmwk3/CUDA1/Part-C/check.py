import torch

H = 1024
W = 1024
C = 3
FW = 3
FH = 3
K = 64
P = 1 

I_0 = torch.empty((C, H+2*P, W+2*P), dtype=torch.float64)
for c in range(C):
    for x in range(H):
        for y in range(W):
            I_0[c, x, y] = c * (x + y)
    for x in range(H, H+2*P):
        for y in range(W, W+2*P):
            I_0[c, x, y] = 0

F = torch.empty((K, C, FH, FW), dtype=torch.float64)
for k in range(K):
    for c in range(C):
        for i in range(FH):
            for j in range(FW):
                F[k, c, i, j] = (c+k) * (i+j)

# from tqdm import tqdm

# O = torch.zeros((K, W, H), dtype=torch.float64)

def getout(k, x, y):
    output = 0.0
    for c in range(C):
        for j in range(FH):
            for i in range(FW):
                output += F[k, c, FW-1-i, FH-1-j] * I_0[c, x+i, y+j]
    
    return output

# with open("output.txt", "w") as f:
#     for k in range(16):
#         for x in range(16):
#             for y in range(16):
#                 result = getout(k, x, y)
#                 f.write(f"{int(result)}")

while True:
    try:
        user_input = input("Enter k, x, y separated by space: ")
        k, x, y = map(int, user_input.split())
        result = getout(k, x, y)
        print(f"Output: {result}")
    except ValueError:
        print("Please enter valid integers for k, x, and y.")
    except KeyboardInterrupt:
        print("\nExiting.")
        break

