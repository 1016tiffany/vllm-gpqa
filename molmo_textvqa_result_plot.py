import matplotlib.pyplot as plt

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 28,
    'axes.linewidth': 1.5,
    'axes.edgecolor': 'black',
    'mathtext.fontset': 'stix',
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.major.size': 6,
    'ytick.major.size': 6
})

# Create figure
plt.figure(figsize=(14, 10), dpi=300)

# Data
n_1bit = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
# Using qwen entropy ordering
# acc = [79.96, 77.46, 76.76, 73.46, 71.92, 70.20, 69.40, 66.12, 66.92, 64.50, 64.30, 63.96, 60.42, 57.36, 54.72]
# Using molmo entropy ordering
# acc = [79.9, 77.46, 75.7, 75.76, 72.46, 71.42, 66.92, 65.18, 64.3, 64.26, 62.86, 60.94, 58.04, 54.42, 54.72]
# Using layer depth ordering, removed load_in_4_bit for the 1-bit model
acc = [79.9, 78.32, 76.76, 70.90, 70.08, 69.64, 69.76, 67.36, 66.62, 64.94, 64.34, 61.20, 59.24, 59.26, 56.74]

# Plot the data
plt.plot(
    n_1bit,
    acc,
    '-o',
    color='red',
    label='Molmo-7B-D-0924',
)

# Customize title and labels
plt.title("TextVQA Accuracy vs. Number of 1-bit Layers", 
          fontsize=30, 
          fontweight='bold', 
          pad=20)
plt.xlabel("Number of 1-bit Layers", 
           fontsize=28, 
           fontweight='bold', 
           labelpad=10)
plt.ylabel("Accuracy (%)", 
           fontsize=28, 
           fontweight='bold', 
           labelpad=10)

# Customize ticks and limits
plt.xticks(ticks=n_1bit, fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(min(n_1bit), max(n_1bit))
plt.ylim(min(acc) - 5, max(acc) + 5)

# Enhance grid
plt.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.2)
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.2)
plt.minorticks_on()

# Customize legend
legend = plt.legend(fontsize=24, 
                    frameon=True, 
                    fancybox=False, 
                    edgecolor='black',
                    loc='best')
legend.get_frame().set_linewidth(1.5)

# Adjust layout
plt.tight_layout()

# Save both PDF and PNG versions
base_path = "./molmo_textvqa_result_plot"
plt.savefig(f"{base_path}.pdf", 
            format='pdf',
            bbox_inches='tight',
            dpi=300,
            pad_inches=0.1)
plt.savefig(f"{base_path}.png",
            format='png',
            bbox_inches='tight',
            dpi=300,
            pad_inches=0.1)
plt.show()

print(f"\nPlots saved as {base_path}.pdf and {base_path}.png")