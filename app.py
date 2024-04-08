from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles 
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException

import json
from typing import List
import torch
import torch.nn as nn  # Import nn module from PyTorch
import matplotlib.pyplot as plt
import numpy as np
import io
import pickle

app = FastAPI()

# Serve the static files directory containing the favicon.ico file
app.mount("/static", StaticFiles(directory="static"), name="static")


# Configure CORS to allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to limit access to specific origins if needed
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


# Define your neural network architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 10000)  # Output tensor shape: (100, 100)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return x.view(-1, 100, 100)  # Reshape the output to the desired 2D tensor shape

# Load your trained model
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Load the scaler from the file
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to perform inference
def inference(values: List[float]) -> torch.Tensor:
    with torch.no_grad():
        # Normalize input values
        values_normalized = scaler.transform([values])  # Assuming values is a list of floats

        # Convert input values to tensor
        input_tensor = torch.tensor(values_normalized, dtype=torch.float32)

        # Perform inference
        output_tensor = model(input_tensor.unsqueeze(0))
   
        # Optionally, inverse transform the output tensor if needed
        # output_tensor = torch.tensor(scaler.inverse_transform(output_tensor), dtype=torch.float32)
    return output_tensor


# Function to plot tensor and save as file
def plot_tensor(tensor: torch.Tensor, filename: str):
    
    tensor_np = tensor.cpu().numpy()   # Convert tensor to numpy array
    plt.rcParams['font.family'] = 'Avenir'   # Set font family for Matplotlib
    
    # Plot tensor
    fig, ax = plt.subplots()   # Create figure
    img = ax.imshow(tensor_np, cmap='hot', origin='lower', vmin=10.0, vmax=150.0)
    plt.colorbar(img, ax=ax, label='Temperature')
    plt.xlabel('Position (x)')
    plt.ylabel('Position (y)')
    plt.title('Temperature Distribution')

    # Save plot as file
    plt.savefig(filename)
    plt.close()


@app.post("/predict/")
async def predict(values: List[float]):
    try:
        # Log the input values received in the request body
        print("Received input values:", values)

        # Perform inference
        output_tensor = inference(values)

        # Convert output tensor to JSON
        output_json = output_tensor.tolist()

        # Return output tensor as JSON response
        return Response(content=json.dumps(output_json), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plot/")
async def predict(values: List[float]):
    # Log the input values received in the request body
    print("Received input values:", values)

    # Perform inference
    output_tensor = inference(values)

    # Plot tensor and save as file
    plot_tensor(output_tensor.squeeze(0), "output.png")

    # Return output file
    return FileResponse("output.png")


@app.get("/")
async def read_root():
    # Return HTML file
    return FileResponse("index.html")