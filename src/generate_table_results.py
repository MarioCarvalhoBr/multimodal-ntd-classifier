import json
import pandas as pd
from pathlib import Path

def generate_summary_table():
    print("[*] Varrendo diretórios de resultados...")
    results_dir = Path("output/results")
    
    if not results_dir.exists():
        print("[!] Diretório 'output/results' não encontrado.")
        return

    data = []

    # Itera sobre todas as pastas dentro de output/results
    for model_dir in results_dir.iterdir():
        if model_dir.is_dir():
            json_path = model_dir / "test_report" / "test_report.json"
            
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        report = json.load(f)
                        
                        # Recupera o nome original do modelo (ex: google_resnet -> google/resnet)
                        model_name = model_dir.name.replace("_", "/", 1) 
                        
                        # Extrai e converte para porcentagem (duas casas decimais)
                        acc = report.get("accuracy", 0.0) * 100
                        f1 = report.get("f1_macro", 0.0) * 100
                        
                        data.append({
                            "Modelo": model_name,
                            "Acurácia (%)": round(acc, 2),
                            "F1-Macro (%)": round(f1, 2)
                        })
                except Exception as e:
                    print(f"[!] Erro ao ler {json_path}: {e}")
            else:
                print(f"[!] Arquivo test_report.json não encontrado para o modelo: {model_dir.name}")

    if not data:
        print("[!] Nenhum dado encontrado para gerar a tabela.")
        return

    # Cria o DataFrame e ordena do melhor para o pior (baseado no F1-Score)
    df = pd.DataFrame(data)
    df = df.sort_values(by="F1-Macro (%)", ascending=False).reset_index(drop=True)

    # 1. Exibe no Terminal de forma bonita
    print("\n" + "="*50)
    print("TABELA DE RESULTADOS FINAIS (CONJUNTO DE TESTE)")
    print("="*50)
    print(df.to_markdown(index=False))
    print("="*50 + "\n")

    # 2. Salva em CSV para abrir no Excel ou planilhas
    csv_path = results_dir / "final_metrics_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[+] Tabela salva em CSV: {csv_path}")

    # 3. Salva a formatação em LaTeX pronta para o artigo
    latex_path = results_dir / "final_metrics_summary.tex"
    latex_code = df.to_latex(index=False, float_format="%.2f", 
                             caption="Resultados de classificação no conjunto de teste para as diferentes arquiteturas multimodais e de visão.",
                             label="tab:resultados_teste",
                             column_format="lcc")
    
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    print(f"[+] Tabela salva em LaTeX: {latex_path}")

if __name__ == "__main__":
    generate_summary_table()