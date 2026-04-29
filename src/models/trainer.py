import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from pathlib import Path

from utils.logger import logger


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
        
        # Filtrar parâmetros com requires_grad = True.
        # Parâmetros com requires_grad = False dentro do Adam podem causar falhas no PyTorch/timm.
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
        
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def save_curves(self, model_name):
        logger.info(f"[*] Salvando curvas de aprendizado para: {model_name}")
        model_tag = model_name.replace("/", "_")
        output_dir = Path(f"output/results/{model_tag}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        # Plot Loss
        plt.figure(figsize=(6, 5))
        plt.plot(epochs, self.history["train_loss"], 'b-', label='Treino')
        plt.plot(epochs, self.history["val_loss"], 'r-', label='Validação')
        plt.title(f'Loss - {model_tag}')
        plt.xlabel('Épocas')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curve.pdf")
        plt.close()
        
        # Plot Accuracy
        plt.figure(figsize=(6, 5))
        plt.plot(epochs, self.history["train_acc"], 'b-', label='Treino')
        plt.plot(epochs, self.history["val_acc"], 'r-', label='Validação')
        plt.title(f'Acurácia - {model_tag}')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_curve.pdf")
        plt.close()
        
        # Save JSON history
        import json
        with open(output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=4)
            
        logger.info(f"[+] Métricas, histórico e gráficos salvos em: {output_dir}")
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
        
        logger.info(f"[*] Iniciando treinamento por {epochs} épocas no device: {self.device}")

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Executa o treino e a validação
            train_loss, train_acc, train_f1 = self._run_epoch(train_loader, is_train=True)
            val_loss, val_acc, val_f1 = self._run_epoch(val_loader, is_train=False)

            # --> ADICIONE ESTAS LINHAS AQUI <--
            # Salva as métricas no histórico para gerar os gráficos depois
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            # ----------------------------------

            logger.info(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
            logger.info(f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

            # Salva o modelo se a loss de validação melhorar
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"[+] Melhor modelo salvo em: {save_path}")
    def test_and_report(self, test_loader: DataLoader, target_names: list):
        """Gera o relatório final de classificação formatado para artigos em uma única passagem."""
        logger.info("\n[*] Avaliando o conjunto de Teste...")
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(test_loader, desc="Evaluating Test Set", leave=False)

        with torch.no_grad():
            for pixel_values, labels in progress_bar:
                pixel_values = pixel_values.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(pixel_values)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * pixel_values.size(0)

                # Predições
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Cálculos Finais
        test_loss = total_loss / len(test_loader.dataset)
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average='macro')

        logger.info("\n" + "="*50)
        logger.info("MÉTRICAS FINAIS PARA O ARTIGO (TEST SET)")
        logger.info("="*50)
        logger.info(f"Loss: {test_loss:.4f} | Acurácia: {test_acc:.4f} | F1-Macro: {test_f1:.4f}\n")
        logger.info(classification_report(all_labels, all_preds, target_names=target_names))