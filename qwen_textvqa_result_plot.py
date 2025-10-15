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
acc = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
n_1bit = [85.62, 80.58, 81.42, 80.66, 77.16, 77.42, 75.16, 70.68, 70.38, 69.92, 58.28, 58.4, 48.34, 37.5, 26.26]

# Plot the data
plt.plot(
    acc,
    n_1bit,
    '-o',
    color='red',
    label='Qwen2.5 VL-7B'
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
plt.xticks(ticks=acc, fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(min(acc), max(acc))
plt.ylim(min(n_1bit) - 5, max(n_1bit) + 5)

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
base_path = "./qwen_textvqa_result_plot"
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