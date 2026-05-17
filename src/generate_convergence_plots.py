import json
import matplotlib.pyplot as plt
from pathlib import Path
# Arg parser
import argparse

parser = argparse.ArgumentParser(description="Generate convergence plots for training experiments.")
parser.add_argument("--results_dir", type=str, default="output/results", help="Directory containing the results.")
parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Split to plot.")

def plot_training_convergence():
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    models_data = {}
    
    print(f"[*] Configurações:")
    print(f"    - Diretório de Resultados: {results_dir}")
    print(f"    - Split para Plotagem: {args.split}")
    
    title_map = {
        "train": "Training",
        "val": "Validation"
    }
    print(f"[*] Gerando gráficos de convergência para: {title_map[args.split]}")

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
        epochs = range(1, len(history[f"{args.split}_loss"]) + 1)
        
        # Gráfico 1: Loss
        ax1.plot(epochs, history[f"{args.split}_loss"], label=label, marker='o', markersize=4, linewidth=1.5, color=color)
        
        # Gráfico 2: Accuracy
        ax2.plot(epochs, history[f"{args.split}_acc"], label=label, marker='s', markersize=4, linewidth=1.5, color=color)

    # Ajustes finos de layout
    ax1.set_title(f'{title_map[args.split]} Loss Convergence', fontsize=14, weight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.set_title(f'{title_map[args.split]} Accuracy Progress', fontsize=14, weight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    
    # Salva os gráficos consolidados
    output_pdf = results_dir / f"{title_map[args.split]}_Convergence_Plots.pdf"
    output_png = results_dir / f"{title_map[args.split]}_Convergence_Plots.png"
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"[+] Gráficos salvos com sucesso em: {output_pdf}")

if __name__ == "__main__":
    plot_training_convergence()