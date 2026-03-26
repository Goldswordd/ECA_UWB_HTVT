# Create Pareto plot and save as PNG

import matplotlib.pyplot as plt

# Data
models = [
    "CNN-LSTM", "FCN-Attention", "CIR-CNN+MLP",
    "SA-TinyML", "LightMamba", "MS-CNN-SA", "ECA-UWB"
]
params = [7441, 200000, 1578, 4627, 25793, 24145, 2374]
accuracy = [88.82, 88.24, 87.96, 93.65, 92.38, 93.10, 91.45]

# Create plot
plt.figure()
plt.xscale('log')

# Plot points
for i, model in enumerate(models):
    if model == "ECA-UWB":
        plt.scatter(params[i], accuracy[i], marker='*')
    else:
        plt.scatter(params[i], accuracy[i])

# Pareto frontier (sorted manually)
pareto_x = [1578, 2374, 4627]
pareto_y = [87.96, 91.45, 93.65]
plt.plot(pareto_x, pareto_y)

# Labels
for i, model in enumerate(models):
    if model in ["ECA-UWB", "SA-TinyML"]:
        plt.text(params[i], accuracy[i], model)

plt.xlabel("Parameter Count (log scale)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Parameter Count (eWINE)")

# Save figure
file_path = "/home/johnw/Documents/Paper/UWB_journals/paper_ecauwb/fig_pareto.png"
plt.savefig(file_path, dpi=300, bbox_inches='tight')
plt.close()

file_path
