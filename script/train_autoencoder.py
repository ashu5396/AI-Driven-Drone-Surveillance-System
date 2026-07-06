import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load CSV
df = pd.read_csv(r"C:\ML_DL_Project\script\drone_behavior_dataset.csv")

#print(df)
features = df[[
    "x",
    "y",
    "speed",
    "inside_zone",
    "hover_time"
]]

#print(features.head())



scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(features)

#print(X_scaled)
#print(len(X_scaled))

X_tensor = torch.FloatTensor(X_scaled)

#print(X_tensor)


class Autoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(

            nn.Linear(5, 4),
            nn.ReLU(),

            nn.Linear(4, 2),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(

            nn.Linear(2, 4),
            nn.ReLU(),

            nn.Linear(4, 5),
            nn.Sigmoid()
        )

    def forward(self, x):

        latent = self.encoder(x)

        reconstructed = self.decoder(latent)

        return reconstructed
model = Autoencoder()

#print(model)
criterion = nn.MSELoss()    
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
epochs = 100

losses = []
model.train()
for epoch in range(epochs):

    # Forward pass
    reconstructed = model(X_tensor)

    loss = criterion(reconstructed, X_tensor)

    # Backpropagation
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    losses.append(loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


 
with torch.no_grad():

    reconstructed = model(X_tensor)

    reconstruction_error = torch.mean(
        (X_tensor - reconstructed) ** 2,
        dim=1
    )

print(reconstruction_error[:10])
'''plt.plot(losses)

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()  '''
threshold = np.percentile(
    reconstruction_error.numpy(),
    95
)

print("Threshold:", threshold)
torch.save(
    model.state_dict(),
    "drone_autoencoder.pth"
)
joblib.dump(
    scaler,
    "scaler.pkl"
)
with open("threshold.txt", "w") as f:
    f.write(str(threshold))
     