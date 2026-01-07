"""
EfficientNet-B0 Early Fusion - Optimized Version
================================================
Best practices for transfer learning with balanced, augmented data.
Includes: EarlyStopping, ReduceLROnPlateau, and optimized hyperparameters.
"""

from tensorflow import keras
import random
keras.utils.set_random_seed(42)

import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Conv2D
)
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.data import Dataset
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef
)

# Per Google Colab - decommenta se necessario
# from google.colab import drive
# drive.mount('/content/drive')

# ==================== CONFIGURAZIONE PARAMETRI OTTIMIZZATI ====================
# Best practices per EfficientNet con transfer learning

# Percorsi
result_folder = "/content/drive/MyDrive/Tesi_magistrale/modelli/ultima_prova_e_basta_di_MRA/"

# ===== BATCH SIZE =====
# Per EfficientNet-B0 con GPU standard (16GB), batch_size 16-32 è ottimale
# Con data augmentation già applicata, 16 è un buon compromesso
batch_size = 16

# ===== EARLY STOPPING - Best Practices =====
# patience: 10-15 epoche per transfer learning (permette convergenza graduale)
# min_delta: 1e-4 per evitare stop prematuro su miglioramenti minimi
patience_early_stop = 10
min_delta = 1e-4

# ===== REDUCE ON PLATEAU - Best Practices =====
# patience: 3-5 epoche (reagisce più velocemente dell'early stopping)
# factor: 0.5 (dimezza il LR, standard per fine-tuning)
# min_lr: 1e-7 (floor per evitare LR troppo bassi)
patience_reduce_lr = 5
reduce_lr_factor = 0.5
min_lr = 1e-7

# ===== EPOCHE PER FASE =====
# Con EarlyStopping e ReduceLROnPlateau, possiamo permettere più epoche
# Il training si fermerà automaticamente quando necessario
epochs_phase1 = 50   # Transfer learning (solo testa) - converge velocemente
epochs_phase2 = 40   # Fine-tuning 30 layer
epochs_phase3 = 40   # Fine-tuning 100 layer
epochs_phase4 = 40   # Fine-tuning completo

# ===== LEARNING RATE - Best Practices per EfficientNet =====
# Fase 1: LR più alto per la testa (layer non pre-trained)
# Fasi 2-4: LR progressivamente più bassi per preservare feature pre-trained
lr_phase1 = 1e-3     # Testa: può imparare velocemente
lr_phase2 = 1e-4     # Fine-tuning iniziale: 10x più basso
lr_phase3 = 5e-5     # Fine-tuning medio
lr_phase4 = 1e-5     # Fine-tuning completo: molto conservativo

# ===== OTTIMIZZATORE =====
# AdamW è preferito per transfer learning (weight decay decoupled)
optimizer_type = 'AdamW'

# ===== WEIGHT DECAY - Best Practices =====
# Per modelli pre-trained: 0.01-0.05 è lo standard
# Con data augmentation già applicata, 0.01 evita over-regularization
weight_decay = 0.01

# ===== DROPOUT =====
# Per EfficientNet (che ha già dropout interno): 0.2-0.3
# Con dati bilanciati e augmentati: 0.3 per robustezza
dropout_rate = 0.3

# ===== LABEL SMOOTHING =====
# Best practice per classificazione: 0.1 migliora generalizzazione
label_smoothing = 0.1

# ===== LAYERS DA SBLOCCARE PER FASE =====
# Progressione graduale per fine-tuning stabile
unfreeze_phase2 = 30    # ~15% del backbone
unfreeze_phase3 = 100   # ~50% del backbone
# Fase 4: tutti i layer

# ===================================================================


def create_callbacks(phase_name, result_folder, patience_es=10, patience_lr=5,
                     min_delta=1e-4, factor=0.5, min_lr=1e-7):
    """
    Crea callbacks ottimizzati per ogni fase di training.

    Best Practices:
    - EarlyStopping: monitora val_loss con patience adeguata
    - ReduceLROnPlateau: riduce LR quando val_loss ristagna
    - ModelCheckpoint: salva il miglior modello per ogni fase
    """

    callbacks = [
        # Early Stopping
        # - monitor='val_loss': più stabile di val_accuracy per classificazione
        # - patience: abbastanza alto da permettere recupero dopo plateau
        # - restore_best_weights: garantisce il miglior modello alla fine
        # - min_delta: evita stop su miglioramenti trascurabili
        EarlyStopping(
            monitor='val_loss',
            patience=patience_es,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),

        # Reduce LR on Plateau
        # - patience più bassa di EarlyStopping (reagisce prima)
        # - factor=0.5: dimezzamento standard, non troppo aggressivo
        # - min_lr: previene LR troppo bassi che bloccano l'apprendimento
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=factor,
            patience=patience_lr,
            min_lr=min_lr,
            verbose=1,
            mode='min'
        ),

        # Model Checkpoint - salva il miglior modello di ogni fase
        ModelCheckpoint(
            filepath=os.path.join(result_folder, f'best_model_{phase_name}.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        )
    ]

    return callbacks


def build_EF_model(input_channels, num_classes, learning_rate, dropout_rate,
                   optimizer_type, weight_decay, label_smoothing=0.1):
    """
    Costruisce il modello EfficientNet-B0 con Early Fusion.

    Best Practices applicate:
    - Conv2D 1x1 per proiezione canali con inizializzazione he_normal
    - Label smoothing per migliorare generalizzazione
    - AdamW per weight decay decoupled
    """
    input_layer = Input(shape=(224, 224, input_channels), name='input_12ch')

    # Proiezione 1x1: 12 canali -> 3 canali
    # kernel_initializer='he_normal' è best practice per layer prima di ReLU
    x = Conv2D(
        filters=3,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer='he_normal',
        name='channel_projection'
    )(input_layer)

    # Backbone EfficientNetB0
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    x = base_model(x)

    # Testa di classificazione
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='head_batch_norm')(x)
    x = Dropout(dropout_rate, name='head_dropout')(x)
    output_layer = Dense(num_classes, activation='softmax', name='output')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer,
                        name='EfficientNetB0_EarlyFusion')

    # Ottimizzatore
    if optimizer_type == 'AdamW':
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Loss con label smoothing
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model, base_model


def unfreeze_model(model, base_model, n_unfreeze, learning_rate,
                   optimizer_type, weight_decay, label_smoothing=0.1):
    """
    Sblocca progressivamente il backbone per fine-tuning.

    Best Practices:
    - BatchNormalization sempre congelata (statistiche pre-computed)
    - LR ridotto per layer pre-trained
    """
    # Congela tutto prima
    for layer in base_model.layers:
        layer.trainable = False

    # Sblocca ultimi n_unfreeze layer (escluso BatchNorm)
    for layer in base_model.layers[-n_unfreeze:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    # Ottimizzatore con nuovo LR
    if optimizer_type == 'AdamW':
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Loss con label smoothing
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    # Stampa layer trainabili per verifica
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"Layer trainabili nel backbone: {trainable_count}/{len(base_model.layers)}")

    return model


def flat_list(L):
    """Appiattisce una lista di liste."""
    flat = []
    for inner_list in L:
        flat.extend(inner_list)
    return flat


def plot_accuracy(history_dict, output_path):
    """Plotta accuracy con annotazioni delle fasi."""
    train_acc = flat_list(history_dict['accuracy'])
    val_acc = flat_list(history_dict['val_accuracy'])
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_acc, linewidth=2, label='Train Accuracy')
    plt.plot(epochs, val_acc, linewidth=2, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy - EfficientNet-B0 Early Fusion', fontsize=16)
    plt.xlabel('Epoca', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot accuracy salvato: {output_path}")


def plot_loss(history_dict, output_path):
    """Plotta loss con annotazioni delle fasi."""
    train_loss = flat_list(history_dict['loss'])
    val_loss = flat_list(history_dict['val_loss'])
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, linewidth=2, label='Train Loss')
    plt.plot(epochs, val_loss, linewidth=2, label='Validation Loss')
    plt.title('Training and Validation Loss - EfficientNet-B0 Early Fusion', fontsize=16)
    plt.xlabel('Epoca', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot loss salvato: {output_path}")


def plot_lr_history(lr_history, output_path):
    """Plotta l'andamento del learning rate durante il training."""
    plt.figure(figsize=(12, 6))
    plt.plot(lr_history, linewidth=2)
    plt.title('Learning Rate Schedule', fontsize=16)
    plt.xlabel('Epoca', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot LR salvato: {output_path}")


# ==================== MAIN TRAINING SCRIPT ====================

if __name__ == "__main__":

    # Carico i dataset
    print("Caricamento dataset...")
    X_train_concat = np.load(os.path.join(result_folder, "X_train_concatenated.npy"), mmap_mode="r")
    X_test_concat = np.load(os.path.join(result_folder, "X_test_concatenated.npy"))
    y_train = np.load(os.path.join(result_folder, "y_train_augmented.npy"))
    y_test = np.load(os.path.join(result_folder, "y_test_original.npy"))

    # Carico gli indici dello split (già generati)
    train_idx = np.load(os.path.join(result_folder, "train_idx.npy"))
    val_idx = np.load(os.path.join(result_folder, "val_idx.npy"))

    # Creo i subset
    X_train_split = X_train_concat[train_idx]
    X_val_split = X_train_concat[val_idx]
    y_train_split = y_train[train_idx]
    y_val_split = y_train[val_idx]

    print(f"Train: {len(train_idx)} campioni")
    print(f"Validation: {len(val_idx)} campioni")
    print(f"Test: {len(y_test)} campioni")
    print(f"Training set shape: {X_train_split.shape}")
    print(f"Validation set shape: {X_val_split.shape}")
    print(f"Test set shape: {X_test_concat.shape}")

    # Creo i tf.data.Dataset
    training_set = (Dataset.from_tensor_slices((X_train_split, y_train_split))
                    .shuffle(len(X_train_split))
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))

    validation_set = (Dataset.from_tensor_slices((X_val_split, y_val_split))
                      .batch(batch_size)
                      .prefetch(tf.data.AUTOTUNE))

    # Dizionario per memorizzare la history completa
    history = {
        "loss": [],
        "val_loss": [],
        "accuracy": [],
        "val_accuracy": [],
        "lr": []  # Tracking del learning rate
    }

    # ==================== FASE 1: Transfer Learning ====================
    print("\n" + "="*70)
    print("FASE 1: Transfer Learning (backbone congelato)")
    print("="*70)

    model, base_model = build_EF_model(
        input_channels=12,
        num_classes=4,
        learning_rate=lr_phase1,
        dropout_rate=dropout_rate,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing
    )

    print(f"Layer totali modello: {len(model.layers)}")
    print(f"Layer backbone EfficientNet: {len(base_model.layers)}")
    print(f"Learning rate: {lr_phase1}")

    callbacks_phase1 = create_callbacks(
        'phase1', result_folder,
        patience_es=patience_early_stop,
        patience_lr=patience_reduce_lr,
        min_delta=min_delta,
        factor=reduce_lr_factor,
        min_lr=min_lr
    )

    history_phase1 = model.fit(
        training_set,
        validation_data=validation_set,
        epochs=epochs_phase1,
        callbacks=callbacks_phase1,
        verbose=1
    )

    history["loss"].append(history_phase1.history["loss"])
    history["val_loss"].append(history_phase1.history["val_loss"])
    history["accuracy"].append(history_phase1.history["accuracy"])
    history["val_accuracy"].append(history_phase1.history["val_accuracy"])

    print("Fase 1 completata.")

    # ==================== FASE 2: Fine-tuning 30 layer ====================
    print("\n" + "="*70)
    print(f"FASE 2: Fine-tuning (ultimi {unfreeze_phase2} layer)")
    print("="*70)

    unfreeze_model(model, base_model, n_unfreeze=unfreeze_phase2,
                   learning_rate=lr_phase2, optimizer_type=optimizer_type,
                   weight_decay=weight_decay, label_smoothing=label_smoothing)

    print(f"Learning rate: {lr_phase2}")

    callbacks_phase2 = create_callbacks(
        'phase2', result_folder,
        patience_es=patience_early_stop,
        patience_lr=patience_reduce_lr,
        min_delta=min_delta,
        factor=reduce_lr_factor,
        min_lr=min_lr
    )

    history_phase2 = model.fit(
        training_set,
        validation_data=validation_set,
        epochs=epochs_phase2,
        callbacks=callbacks_phase2,
        verbose=1
    )

    history["loss"].append(history_phase2.history["loss"])
    history["val_loss"].append(history_phase2.history["val_loss"])
    history["accuracy"].append(history_phase2.history["accuracy"])
    history["val_accuracy"].append(history_phase2.history["val_accuracy"])

    print("Fase 2 completata.")

    # ==================== FASE 3: Fine-tuning 100 layer ====================
    print("\n" + "="*70)
    print(f"FASE 3: Fine-tuning (ultimi {unfreeze_phase3} layer)")
    print("="*70)

    unfreeze_model(model, base_model, n_unfreeze=unfreeze_phase3,
                   learning_rate=lr_phase3, optimizer_type=optimizer_type,
                   weight_decay=weight_decay, label_smoothing=label_smoothing)

    print(f"Learning rate: {lr_phase3}")

    callbacks_phase3 = create_callbacks(
        'phase3', result_folder,
        patience_es=patience_early_stop,
        patience_lr=patience_reduce_lr,
        min_delta=min_delta,
        factor=reduce_lr_factor,
        min_lr=min_lr
    )

    history_phase3 = model.fit(
        training_set,
        validation_data=validation_set,
        epochs=epochs_phase3,
        callbacks=callbacks_phase3,
        verbose=1
    )

    history["loss"].append(history_phase3.history["loss"])
    history["val_loss"].append(history_phase3.history["val_loss"])
    history["accuracy"].append(history_phase3.history["accuracy"])
    history["val_accuracy"].append(history_phase3.history["val_accuracy"])

    print("Fase 3 completata.")

    # ==================== FASE 4: Fine-tuning completo ====================
    print("\n" + "="*70)
    print("FASE 4: Fine-tuning completo (tutti i layer)")
    print("="*70)

    unfreeze_model(model, base_model, n_unfreeze=len(base_model.layers),
                   learning_rate=lr_phase4, optimizer_type=optimizer_type,
                   weight_decay=weight_decay, label_smoothing=label_smoothing)

    print(f"Learning rate: {lr_phase4}")

    callbacks_phase4 = create_callbacks(
        'phase4', result_folder,
        patience_es=patience_early_stop,
        patience_lr=patience_reduce_lr,
        min_delta=min_delta,
        factor=reduce_lr_factor,
        min_lr=min_lr
    )

    history_phase4 = model.fit(
        training_set,
        validation_data=validation_set,
        epochs=epochs_phase4,
        callbacks=callbacks_phase4,
        verbose=1
    )

    history["loss"].append(history_phase4.history["loss"])
    history["val_loss"].append(history_phase4.history["val_loss"])
    history["accuracy"].append(history_phase4.history["accuracy"])
    history["val_accuracy"].append(history_phase4.history["val_accuracy"])

    print("\n" + "="*70)
    print("TRAINING COMPLETATO!")
    print("="*70)

    # Salvo il modello finale
    model.save(os.path.join(result_folder, "final_model_optimized.keras"))
    print(f"Modello salvato: {os.path.join(result_folder, 'final_model_optimized.keras')}")

    # Salvo la history e genero i grafici
    np.save(os.path.join(result_folder, "training_history_optimized.npy"), history)
    plot_loss(history, os.path.join(result_folder, "loss_plot_optimized.png"))
    plot_accuracy(history, os.path.join(result_folder, "accuracy_plot_optimized.png"))

    # ==================== VALUTAZIONE SUL TEST SET ====================
    print("\n" + "="*70)
    print("VALUTAZIONE SUL TEST SET")
    print("="*70)

    test_loss, test_acc = model.evaluate(X_test_concat, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Predizioni
    y_pred_prob = model.predict(X_test_concat, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    class_names = ['None', 'Slight', 'Moderate', 'Heavy']

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)

    with open(os.path.join(result_folder, "classification_report_test_optimized.txt"), "w", encoding="utf-8") as f:
        f.write("Classification Report - Test Set (Early Fusion Optimized)\n")
        f.write("="*70 + "\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  - Batch size: {batch_size}\n")
        f.write(f"  - Dropout: {dropout_rate}\n")
        f.write(f"  - Weight decay: {weight_decay}\n")
        f.write(f"  - Label smoothing: {label_smoothing}\n")
        f.write(f"  - Optimizer: {optimizer_type}\n")
        f.write(f"  - LR phases: {lr_phase1}, {lr_phase2}, {lr_phase3}, {lr_phase4}\n")
        f.write(f"  - Early stopping patience: {patience_early_stop}\n")
        f.write(f"  - ReduceLROnPlateau patience: {patience_reduce_lr}\n")
        f.write("="*70 + "\n\n")
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - EfficientNet-B0 Early Fusion (Optimized)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, "confusion_matrix_EF_optimized.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # Metriche
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc_ovo = roc_auc_score(y_test, y_pred_prob, multi_class='ovo')
    mcc = matthews_corrcoef(y_true, y_pred)

    print("\nMetriche finali:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc_ovo:.4f}")
    print(f"  MCC:       {mcc:.4f}")

    # Salvo le metriche
    with open(os.path.join(result_folder, "metrics_optimized.txt"), "w", encoding="utf-8") as f:
        f.write("Metriche di Valutazione - Test Set (Early Fusion Optimized)\n")
        f.write("="*70 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"AUC-ROC:   {auc_ovo:.4f}\n")
        f.write(f"MCC:       {mcc:.4f}\n")

    print(f"\nTutti i risultati salvati in: {result_folder}")
