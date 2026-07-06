import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import joblib
df = pd.read_csv(
    r"C:\ML_DL_Project\script\drone_behavior_dataset.csv"
)
feature_columns = [
    "x",
    "y",
    "speed",
    "hover_time"
]
scaler = MinMaxScaler()

df[feature_columns] = scaler.fit_transform(
    df[feature_columns]
)

joblib.dump(
    scaler,
    "lstm_scaler.pkl"
)
SEQ_LENGTH = 10

all_sequences = []

# GROUP BY CAMERA + DRONE
grouped = df.groupby(
    ["cam_id", "drone_id"]
)

# PROCESS EACH TRAJECTORY SEPARATELY
for (cam_id, drone_id), group in grouped:

    # SORT BY TIME
    group = group.sort_values(
        by="timestamp"
    ).reset_index(drop=True)

    # EXTRACT FEATURES
    data = group[feature_columns].values
    '''print("\n======================")
    print(f"Camera ID: {cam_id}")
    print(f"Drone ID: {drone_id}")

    print("\nTrajectory Data:")
    print(data[:15])'''

    # SKIP SHORT TRACKS
    if len(data) < SEQ_LENGTH:
        continue

    # CREATE TEMPORAL WINDOWS
    for i in range(len(data) - SEQ_LENGTH):
        if i == 1:

            print("\nFirst Sequence:")
            print(data[i:i + SEQ_LENGTH])

        seq = data[i:i + SEQ_LENGTH]
        
        all_sequences.append(seq)
        
       

# FINAL NUMPY ARRAY
if len(all_sequences) == 0:
    print("No valid sequences found.")
    exit()
X_sequences = np.array(all_sequences)
print(X_sequences.shape)


X_tensor = torch.FloatTensor(
    X_sequences
)
print( "inside one sequence:- ", X_tensor[0])
class LSTMAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.LSTM(
            input_size=4,
            hidden_size=16,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=16,
            hidden_size=4,
            batch_first=True
        )

    def forward(self, x):

        encoded, (hidden, cell) = self.encoder(x)

        reconstructed, _ = self.decoder(encoded)

        return reconstructed
model = LSTMAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
epochs = 50
train_losses = []

for epoch in range(epochs):

    reconstructed = model(X_tensor)

    loss = criterion(
        reconstructed,
        X_tensor
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
    train_losses.append(loss.item())

    if epoch % 5 == 0:

        print(
            f"Epoch {epoch}, "
            f"Loss: {loss.item():.6f}"
        )
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(train_losses, linewidth=2)

plt.xlabel("Epoch")

plt.ylabel("Training Loss (MSE)")

plt.title("Training Loss of LSTM Autoencoder")

plt.grid(True)

plt.savefig("lstm_training_loss.png", dpi=300)

plt.show()        
with torch.no_grad():

    reconstructed = model(X_tensor)

    reconstruction_error = torch.mean(
        (X_tensor - reconstructed) ** 2,
        dim=(1,2)
    ) 
errors = reconstruction_error.numpy()

plt.figure(figsize=(8,5))

plt.hist(errors,
         bins=40,
         edgecolor='black')

plt.xlabel("Reconstruction Error")

plt.ylabel("Number of Sequences")

plt.title("Distribution of Reconstruction Errors")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Figure_5_9_Reconstruction_Error.png",
    dpi=300
)

plt.show()         
threshold = np.percentile(
    errors,
    95
)
# ==========================
# FIGURE 5.10
# Threshold Visualization
# ==========================

plt.figure(figsize=(8,5))

plt.hist(
    errors,
    bins=40,
    edgecolor='black'
)

plt.axvline(
    threshold,
    color='red',
    linestyle='--',
    linewidth=3,
    label='95th Percentile Threshold'
)

plt.xlabel("Reconstruction Error")
plt.ylabel("Number of Sequences")
plt.title("Threshold Selection for Anomaly Detection")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Figure_5_10_Threshold.png",
    dpi=300
)

plt.show()

# ==========================
# FIGURE 5.11
# Normal vs Anomaly
# ==========================

labels = errors > threshold

colors = []

for label in labels:

    if label:
        colors.append("red")
    else:
        colors.append("blue")

plt.figure(figsize=(10,5))

plt.scatter(
    range(len(errors)),
    errors,
    c=colors,
    s=20
)

plt.axhline(
    threshold,
    color='green',
    linestyle='--',
    linewidth=2,
    label='Threshold'
)

plt.xlabel("Sequence Number")
plt.ylabel("Reconstruction Error")
plt.title("Normal vs Anomalous Drone Trajectories")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Figure_5_11_Normal_vs_Anomaly.png",
    dpi=300
)

plt.show()
# ==========================
# FIGURE 5.12
# Reconstruction Error Trend
# ==========================

plt.figure(figsize=(10,5))

plt.plot(
    errors,
    linewidth=2
)

plt.axhline(
    threshold,
    color='red',
    linestyle='--',
    linewidth=2,
    label='Threshold'
)

plt.xlabel("Sequence Number")
plt.ylabel("Reconstruction Error")

plt.title("Reconstruction Error Across All Sequences")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Figure_5_12_Error_Trend.png",
    dpi=300
)

plt.show()

print("Threshold:", threshold)
with open("lstm_threshold.txt", "w") as f:

    f.write(str(threshold))
torch.save(
    model.state_dict(),
    "lstm_autoencoder.pth"
)      