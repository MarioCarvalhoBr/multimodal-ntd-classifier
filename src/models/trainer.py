import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from pathlib import Path

class ModelTrainer:
    """
    Orquestra o ciclo de vida de treinamento, validação e teste.
    Calcula métricas rigorosas para artigos científicos.
    """
    def __init__(self, model, device, lr=1e-3):
        self.model = model
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        # Otimizador criado aqui garante que os parâmetros já estejam no device correto
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def save_curves(self, model_name):
        print(f"[*] Salvando curvas de aprendizado para: {model_name}")
        output_dir = Path("output/figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        epochs = range(1, len(self.history["train_loss"]) + 1)
        model_tag = model_name.replace("/", "_")
        
        plt.figure(figsize=(12, 5))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history["train_loss"], 'b-', label='Treino')
        plt.plot(epochs, self.history["val_loss"], 'r-', label='Validação')
        plt.title(f'Loss - {model_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history["train_acc"], 'b-', label='Treino')
        plt.plot(epochs, self.history["val_acc"], 'r-', label='Validação')
        plt.title(f'Acurácia - {model_name}')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f"learning_curves_{model_tag}.pdf")
        plt.close()
        print(f"[+] Curvas salvas em: {output_dir / f'learning_curves_{model_tag}.pdf'}")
    def _run_epoch(self, dataloader: DataLoader, is_train: bool = True):
        """Método interno para rodar uma época (Treino ou Validação)."""
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        # Barra de progresso
        progress_bar = tqdm(dataloader, desc="Training" if is_train else "Evaluating", leave=False)

        for pixel_values, labels in progress_bar:
            pixel_values = pixel_values.to(self.device)
            labels = labels.to(self.device)

            # Forward pass com cálculo de gradiente apenas se for treino
            with torch.set_grad_enabled(is_train):
                outputs = self.model(pixel_values)
                loss = self.criterion(outputs, labels)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item() * pixel_values.size(0)
            
            # Obtendo as predições
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Cálculo de métricas da época
        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='macro')

        return epoch_loss, epoch_acc, epoch_f1

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, save_path: str):
        """Loop principal de treinamento com salvamento do melhor modelo."""
        best_val_loss = float('inf')
        
        print(f"[*] Iniciando treinamento por {epochs} épocas no device: {self.device}")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_acc, train_f1 = self._run_epoch(train_loader, is_train=True)
            val_loss, val_acc, val_f1 = self._run_epoch(val_loader, is_train=False)

            print(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
            print(f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

            # Salva o modelo se a loss de validação melhorar
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"[+] Melhor modelo salvo em: {save_path}")

    def test_and_report(self, test_loader: DataLoader, target_names: list):
        """Gera o relatório final de classificação formatado para artigos."""
        print("\n[*] Avaliando o conjunto de Teste...")
        test_loss, test_acc, test_f1 = self._run_epoch(test_loader, is_train=False)
        
        # Coletar predições finais para o classification report detalhado
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for pixel_values, labels in test_loader:
                outputs = self.model(pixel_values.to(self.device))
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        print("\n" + "="*50)
        print("MÉTRICAS FINAIS PARA O ARTIGO (TEST SET)")
        print("="*50)
        print(f"Loss: {test_loss:.4f} | Acurácia: {test_acc:.4f} | F1-Macro: {test_f1:.4f}\n")
        print(classification_report(all_labels, all_preds, target_names=target_names))