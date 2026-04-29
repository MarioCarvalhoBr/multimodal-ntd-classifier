import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np
import sys

# Adiciona a pasta 'src' ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.preprocessors import HairRemovalFilter

def generate_eda_reports():
    data_dir = Path("data/processed/Dataset-NTD-V1")
    output_dir = Path("output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Coleta de dados para os gráficos de frequência
    records = []
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        for cls in split_dir.iterdir():
            if cls.is_dir():
                count = len(list(cls.glob("*.*")))
                records.append({"Split": split, "Classe": cls.name, "Quantidade": count})
    
    df_counts = pd.DataFrame(records)
    sns.set_theme(style="whitegrid")

    # --- Gráfico de Barras Empilhadas (Panorama Geral) ---
    plt.figure(figsize=(12, 6))
    pivot_df = df_counts.pivot(index='Classe', columns='Split', values='Quantidade')
    pivot_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
    plt.title("Distribuição Geral de Classes por Partição")
    plt.ylabel("Número de Imagens")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution_bars.pdf")

    # --- Gráfico de Pizza (Proporção Total do Dataset) ---
    plt.figure(figsize=(8, 8))
    total_counts = df_counts.groupby('Classe')['Quantidade'].sum()
    plt.pie(total_counts, labels=total_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Pastel1"))
    plt.title("Proporção Total das Classes no Dataset-NTD-V1")
    plt.savefig(output_dir / "class_distribution_pie.pdf")

    # --- Gráfico de Violino (Densidade por Split) ---
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_counts, x='Split', y='Quantidade', hue='Split', palette='muted', legend=False)
    plt.title("Densidade da Quantidade de Arquivos por Partição")
    plt.savefig(output_dir / "split_density_violin.pdf")

    # 2. Demonstração de Remoção de Pelos (Antes e Depois)
    # Selecionamos 10 casos da classe clínica (Hanseníase)
    clinical_dir = data_dir / "train/clinical_leprosy"
    sample_images = list(clinical_dir.glob("*.jpg"))[:10]
    
    if sample_images:
        preprocessor = HairRemovalFilter()
        fig, axes = plt.subplots(10, 2, figsize=(10, 30))
        plt.suptitle("Demonstração: Remoção de Pelos (Black-Hat + Inpainting)", fontsize=16)

        for i, img_path in enumerate(sample_images):
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed = preprocessor.process(img_rgb)
            
            axes[i, 0].imshow(img_rgb)
            axes[i, 0].set_title(f"Original {i+1}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(processed)
            axes[i, 1].set_title(f"Processada {i+1}")
            axes[i, 1].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_dir / "hair_removal_demo.pdf")

    print(f"[+] Gráficos gerados com sucesso em: {output_dir}")

if __name__ == "__main__":
    generate_eda_reports()