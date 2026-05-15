import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_convergence():
    results_dir = Path("output/results")
    models_data = {}

    print("[*] Lendo arquivos history.json...")
    for model_dir in results_dir.iterdir():
        if model_dir.is_dir():
            history_path = model_dir / "history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    # Limpa o nome do modelo para a legenda (ex: google_vit -> ViT-B/16)
                    label = model_dir.name.replace("_", "/").replace("google/", "").replace("openai/", "").replace("microsoft/", "")
                    models_data[label] = json.load(f)

    if not models_data:
        print("[!] Nenhum dado de histórico encontrado.")
        return

    # Configuração estética para o Artigo (SBC Style)
    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for (label, history), color in zip(models_data.items(), colors):
        epochs = range(1, len(history["val_loss"]) + 1)
        
        # Gráfico 1: Val Loss
        ax1.plot(epochs, history["val_loss"], label=label, marker='o', markersize=4, linewidth=1.5, color=color)
        
        # Gráfico 2: Val Accuracy
        ax2.plot(epochs, history["val_acc"], label=label, marker='s', markersize=4, linewidth=1.5, color=color)

    # Ajustes finos de layout
    ax1.set_title('Validation Loss Convergence', fontsize=14, weight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.set_title('Validation Accuracy Progress', fontsize=14, weight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    
    # Salva os gráficos consolidados
    output_pdf = results_dir / "models_convergence_comparison.pdf"
    output_png = results_dir / "models_convergence_comparison.png"
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"[+] Gráficos salvos com sucesso em: {output_pdf}")

if __name__ == "__main__":
    plot_training_convergence()