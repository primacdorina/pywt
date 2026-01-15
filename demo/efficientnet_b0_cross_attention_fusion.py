# ============================================================================
# EfficientNet-B0 Cross-Attention Fusion - Complete Training Pipeline
# ============================================================================
# Architettura:
#   - Branch RGB: Input(224,224,3) → EfficientNetB0 (ImageNet) → FeatureMap(7,7,1280)
#   - Branch MRA: Input(224,224,9) → Conv1x1(9→3) → EfficientNetB0 (ImageNet) → FeatureMap(7,7,1280)
#   - Cross-Attention: RGB ↔ MRA (bidirezionale sui 49 "patch" delle feature maps)
#   - Fusione: Concatenate → Dropout → Dense(4, softmax)
# ============================================================================

from tensorflow import keras
import random
keras.utils.set_random_seed(42)

import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import gc
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Conv2D, concatenate,
    MultiHeadAttention, LayerNormalization, Add, Reshape, Layer
)
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.data import Dataset
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef
)

# Per Google Colab: decommentare le righe seguenti
# from google.colab import drive
# drive.mount('/content/drive')


# ============================================================================
# CONFIGURAZIONE PARAMETRI
# ============================================================================

# Percorsi (modifica secondo la tua struttura)
base_dir = "/content/drive/MyDrive/Tesi_magistrale/modelli/ultima_prova_e_basta_di_MRA/"
rgb_dir = "/content/drive/MyDrive/Tesi_magistrale/modelli/"
result_folder = "/content/drive/MyDrive/Tesi_magistrale/modelli/cross_attention_fusion/"

# Crea cartella risultati se non esiste
os.makedirs(result_folder, exist_ok=True)

# Batch size
batch_size = 32

# Early Stopping
patience_early_stop = 8

# Reduce LR on Plateau
patience_reduce_lr = 3
reduce_lr_factor = 0.5
min_lr = 1e-7

# Epoche per fase
epochs_phase1 = 50
epochs_phase2 = 40
epochs_phase3 = 40
epochs_phase4 = 40

# Learning rate per fase
lr_phase1 = 1e-3
lr_phase2 = 1e-5
lr_phase3 = 1e-5
lr_phase4 = 1e-5

# Optimizer
optimizer_type = 'AdamW'
weight_decay = 1e-4
dropout_rate = 0.25

# Cross-Attention config
num_attention_heads = 4  # Numero di teste per MultiHeadAttention
attention_dropout = 0.1  # Dropout nell'attention

# Layer da sbloccare per fase (per CIASCUN backbone)
unfreeze_phase2 = 30
unfreeze_phase3 = 130


# ============================================================================
# CARICAMENTO DATASET
# ============================================================================

print("Caricamento dataset...")
X_train_rgb = np.load(os.path.join(rgb_dir, "X_train_augmented.npy"), mmap_mode="r")
X_train_mra = np.load(os.path.join(base_dir, "X_train_mra_details.npy"), mmap_mode="r")
y_train = np.load(os.path.join(rgb_dir, "y_train_augmented.npy"))

X_test_rgb = np.load(os.path.join(rgb_dir, "X_test_original.npy"))
X_test_mra = np.load(os.path.join(base_dir, "X_test_mra_details.npy"))
y_test = np.load(os.path.join(rgb_dir, "y_test_original.npy"))

print(f"RGB train shape: {X_train_rgb.shape}")
print(f"MRA train shape: {X_train_mra.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"RGB test shape: {X_test_rgb.shape}")
print(f"MRA test shape: {X_test_mra.shape}")
print(f"y_test shape: {y_test.shape}")


# ============================================================================
# CREAZIONE DATASET TF.DATA
# ============================================================================

def create_dataset(X_rgb, X_mra, y, batch_size, shuffle=True):
    """Crea un tf.data.Dataset per training/validation."""
    dataset = Dataset.from_tensor_slices(((X_rgb, X_mra), y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(y), seed=42)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Split training/validation (90/10)
n_samples = len(y_train)
n_val = int(n_samples * 0.1)
indices = np.random.permutation(n_samples)
val_indices = indices[:n_val]
train_indices = indices[n_val:]

# Nota: con mmap_mode="r" dobbiamo copiare i dati per la validation
X_val_rgb = np.array(X_train_rgb[val_indices])
X_val_mra = np.array(X_train_mra[val_indices])
y_val = y_train[val_indices]

print(f"\nTraining samples: {len(train_indices)}")
print(f"Validation samples: {len(val_indices)}")

# Crea i dataset
training_set = create_dataset(
    np.array(X_train_rgb[train_indices]),
    np.array(X_train_mra[train_indices]),
    y_train[train_indices],
    batch_size, shuffle=True
)
validation_set = create_dataset(X_val_rgb, X_val_mra, y_val, batch_size, shuffle=False)


# ============================================================================
# CROSS-ATTENTION BLOCK
# ============================================================================

class CrossAttentionBlock(Layer):
    """
    Blocco di Cross-Attention bidirezionale.

    RGB features attendono a MRA features e viceversa.
    Include residual connections e layer normalization.
    """

    def __init__(self, num_heads=4, key_dim=None, dropout=0.1, name_prefix='cross_attn', **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
        self.name_prefix = name_prefix

    def build(self, input_shape):
        # input_shape è una lista di due shapes: [rgb_shape, mra_shape]
        # Assumiamo che siano uguali: (batch, seq_len, features)
        feature_dim = input_shape[0][-1]

        if self.key_dim is None:
            self.key_dim = feature_dim // self.num_heads

        # Cross-Attention: RGB → MRA (RGB query, MRA key/value)
        self.cross_attn_rgb_to_mra = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate,
            name=f'{self.name_prefix}_rgb_to_mra'
        )

        # Cross-Attention: MRA → RGB (MRA query, RGB key/value)
        self.cross_attn_mra_to_rgb = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate,
            name=f'{self.name_prefix}_mra_to_rgb'
        )

        # Layer Normalization
        self.ln_rgb = LayerNormalization(name=f'{self.name_prefix}_ln_rgb')
        self.ln_mra = LayerNormalization(name=f'{self.name_prefix}_ln_mra')

        # Dropout per residual
        self.dropout_rgb = Dropout(self.dropout_rate)
        self.dropout_mra = Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: lista [feat_rgb, feat_mra], ciascuno di shape (batch, seq_len, features)
            training: flag per dropout

        Returns:
            lista [feat_rgb_enhanced, feat_mra_enhanced]
        """
        feat_rgb, feat_mra = inputs

        # Cross-Attention RGB → MRA
        # RGB fa query, MRA fornisce key e value
        # "Cosa nelle features MRA è rilevante per RGB?"
        attn_rgb = self.cross_attn_rgb_to_mra(
            query=feat_rgb,
            key=feat_mra,
            value=feat_mra,
            training=training
        )
        attn_rgb = self.dropout_rgb(attn_rgb, training=training)
        feat_rgb_enhanced = self.ln_rgb(feat_rgb + attn_rgb)

        # Cross-Attention MRA → RGB
        # MRA fa query, RGB fornisce key e value
        # "Cosa nelle features RGB è rilevante per MRA?"
        attn_mra = self.cross_attn_mra_to_rgb(
            query=feat_mra,
            key=feat_rgb,
            value=feat_rgb,
            training=training
        )
        attn_mra = self.dropout_mra(attn_mra, training=training)
        feat_mra_enhanced = self.ln_mra(feat_mra + attn_mra)

        return [feat_rgb_enhanced, feat_mra_enhanced]

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout_rate,
            'name_prefix': self.name_prefix
        })
        return config


# ============================================================================
# BUILD MODEL CON CROSS-ATTENTION
# ============================================================================

def build_cross_attention_model(num_classes, learning_rate, dropout_rate,
                                 optimizer_type, weight_decay,
                                 num_heads=4, attention_dropout=0.1):
    """
    Costruisce il modello EfficientNet-B0 con Cross-Attention Fusion.

    Architettura:
    1. Due branch EfficientNet-B0 (RGB e MRA) estraggono feature maps (7x7x1280)
    2. Le feature maps vengono reshapate in sequenze (49x1280) - 49 "patch"
    3. Cross-Attention bidirezionale permette ai branch di comunicare
    4. Global Average Pooling e concatenazione per la classificazione finale
    """

    print("=" * 70)
    print("COSTRUZIONE MODELLO CON CROSS-ATTENTION")
    print("=" * 70)

    # Carica pesi ImageNet una volta sola
    print("Caricamento pesi ImageNet...")
    temp_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    imagenet_weights = temp_model.get_weights()
    del temp_model
    gc.collect()

    # ===== INPUT LAYERS =====
    input_rgb = Input(shape=(224, 224, 3), name='input_rgb')
    input_mra = Input(shape=(224, 224, 9), name='input_mra')

    # ===== BRANCH RGB =====
    print("Creazione branch RGB...")
    base_model_rgb = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3),
        name='efficientnet_rgb'
    )
    base_model_rgb.set_weights(imagenet_weights)
    base_model_rgb.trainable = False  # Congelato inizialmente

    # Feature maps RGB: (batch, 7, 7, 1280)
    feat_rgb = base_model_rgb(input_rgb)

    # ===== BRANCH MRA =====
    print("Creazione branch MRA...")
    # Proiezione 9 canali → 3 canali per compatibilità con EfficientNet
    mra_projected = Conv2D(
        filters=3,
        kernel_size=(1, 1),
        padding='same',
        name='mra_channel_projection'
    )(input_mra)

    base_model_mra = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3),
        name='efficientnet_mra'
    )
    base_model_mra.set_weights(imagenet_weights)
    base_model_mra.trainable = False  # Congelato inizialmente

    # Feature maps MRA: (batch, 7, 7, 1280)
    feat_mra = base_model_mra(mra_projected)

    # ===== RESHAPE PER ATTENTION =====
    # Da (batch, 7, 7, 1280) a (batch, 49, 1280)
    # Ogni posizione 7x7 diventa un "token" / "patch"
    print("Preparazione sequenze per Cross-Attention...")
    feat_rgb_seq = Reshape((7 * 7, 1280), name='reshape_rgb_to_seq')(feat_rgb)
    feat_mra_seq = Reshape((7 * 7, 1280), name='reshape_mra_to_seq')(feat_mra)

    # ===== CROSS-ATTENTION =====
    print(f"Aggiunta Cross-Attention ({num_heads} heads)...")
    cross_attention = CrossAttentionBlock(
        num_heads=num_heads,
        dropout=attention_dropout,
        name_prefix='cross_attn'
    )
    feat_rgb_enhanced, feat_mra_enhanced = cross_attention([feat_rgb_seq, feat_mra_seq])

    # ===== POOLING =====
    # Global Average Pooling sulle sequenze: media su 49 posizioni
    # Da (batch, 49, 1280) a (batch, 1280)
    x_rgb = tf.keras.layers.GlobalAveragePooling1D(name='gap_rgb')(feat_rgb_enhanced)
    x_mra = tf.keras.layers.GlobalAveragePooling1D(name='gap_mra')(feat_mra_enhanced)

    # Batch Normalization
    x_rgb = BatchNormalization(name='bn_rgb')(x_rgb)
    x_mra = BatchNormalization(name='bn_mra')(x_mra)

    # ===== FUSION + CLASSIFICATION HEAD =====
    print("Creazione head di classificazione...")
    # Concatenazione: (batch, 2560)
    x = concatenate([x_rgb, x_mra], axis=1, name='fusion_concat')

    # Dropout
    x = Dropout(dropout_rate, name='head_dropout')(x)

    # Output layer
    output = Dense(num_classes, activation='softmax', name='output')(x)

    # ===== CREAZIONE MODELLO =====
    model = Model(
        inputs=[input_rgb, input_mra],
        outputs=output,
        name='EfficientNetB0_CrossAttention_Fusion'
    )

    # ===== COMPILAZIONE =====
    if optimizer_type == 'AdamW':
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Summary
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    model.summary()

    print(f"\nTotale parametri: {model.count_params():,}")
    print(f"Parametri trainabili: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

    return model, base_model_rgb, base_model_mra


# ============================================================================
# FUNZIONE UNFREEZE PER FINE-TUNING
# ============================================================================

def unfreeze_model(model, base_model_rgb, base_model_mra, n_unfreeze,
                   learning_rate, optimizer_type, weight_decay):
    """
    Sblocca gli ultimi n_unfreeze layer di ciascun backbone per fine-tuning.
    I layer BatchNormalization rimangono congelati per stabilità.
    """

    # Prima congela tutto
    for layer in base_model_rgb.layers:
        layer.trainable = False
    for layer in base_model_mra.layers:
        layer.trainable = False

    # Sblocca ultimi n_unfreeze layer (escludendo BatchNorm)
    for layer in base_model_rgb.layers[-n_unfreeze:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    for layer in base_model_mra.layers[-n_unfreeze:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    # Conta layer trainabili
    trainable_rgb = sum([1 for l in base_model_rgb.layers if l.trainable])
    trainable_mra = sum([1 for l in base_model_mra.layers if l.trainable])
    print(f"Layer trainabili RGB: {trainable_rgb}/{len(base_model_rgb.layers)}")
    print(f"Layer trainabili MRA: {trainable_mra}/{len(base_model_mra.layers)}")

    # Ricompila con nuovo learning rate
    if optimizer_type == 'AdamW':
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def flat_list(L):
    """Appiattisce una lista di liste."""
    flat = []
    for inner_list in L:
        flat.extend(inner_list)
    return flat


def plot_training_history(history_dict, output_folder, model_name='CrossAttn'):
    """Genera e salva i plot di training."""

    # Plot Accuracy
    train_acc = flat_list(history_dict['accuracy'])
    val_acc = flat_list(history_dict['val_accuracy'])
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'b-', linewidth=2, label='Train Accuracy')
    plt.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Plot Loss
    train_loss = flat_list(history_dict['loss'])
    val_loss = flat_list(history_dict['val_loss'])

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'training_history_{model_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot salvato: training_history_{model_name}.png")


def evaluate_model(model, X_test_rgb, X_test_mra, y_test, output_folder, model_name='CrossAttn'):
    """Valutazione completa del modello sul test set."""

    print("\n" + "=" * 70)
    print("VALUTAZIONE SUL TEST SET")
    print("=" * 70)

    class_names = ['None', 'Slight', 'Moderate', 'Heavy']

    # Predizioni
    test_loss, test_acc = model.evaluate([X_test_rgb, X_test_mra], y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    y_pred_prob = model.predict([X_test_rgb, X_test_mra], verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report)

    with open(os.path.join(output_folder, f'classification_report_{model_name}.txt'), 'w') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("=" * 70 + "\n")
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'confusion_matrix_{model_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Metriche
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc_ovo = roc_auc_score(y_test, y_pred_prob, multi_class='ovo')
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"\nMetriche finali:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc_ovo:.4f}")
    print(f"  MCC:       {mcc:.4f}")

    # Salva metriche
    with open(os.path.join(output_folder, f'metrics_{model_name}.txt'), 'w') as f:
        f.write(f"Metriche di Valutazione - {model_name}\n")
        f.write("=" * 70 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"AUC-ROC:   {auc_ovo:.4f}\n")
        f.write(f"MCC:       {mcc:.4f}\n")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_ovo,
        'mcc': mcc
    }


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_model():
    """Pipeline completa di training in 4 fasi."""

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience_early_stop,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=reduce_lr_factor,
        patience=patience_reduce_lr,
        min_lr=min_lr,
        verbose=1
    )

    callbacks = [early_stop, reduce_lr]

    # History per tutte le fasi
    history = {
        "loss": [], "val_loss": [],
        "accuracy": [], "val_accuracy": []
    }

    # ==================== BUILD MODEL ====================
    model, base_model_rgb, base_model_mra = build_cross_attention_model(
        num_classes=4,
        learning_rate=lr_phase1,
        dropout_rate=dropout_rate,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
        num_heads=num_attention_heads,
        attention_dropout=attention_dropout
    )

    # ==================== FASE 1: Transfer Learning ====================
    print("\n" + "=" * 70)
    print("FASE 1: Transfer Learning (backbone congelati)")
    print("=" * 70)
    print("Solo head di classificazione e Cross-Attention trainabili")

    history_phase1 = model.fit(
        training_set,
        validation_data=validation_set,
        epochs=epochs_phase1,
        callbacks=callbacks,
        verbose=1
    )

    history["loss"].append(history_phase1.history["loss"])
    history["val_loss"].append(history_phase1.history["val_loss"])
    history["accuracy"].append(history_phase1.history["accuracy"])
    history["val_accuracy"].append(history_phase1.history["val_accuracy"])
    print("Fase 1 completata.")

    # ==================== FASE 2: Fine-tuning parziale ====================
    print("\n" + "=" * 70)
    print(f"FASE 2: Fine-tuning (ultimi {unfreeze_phase2} layer per backbone)")
    print("=" * 70)

    model = unfreeze_model(
        model, base_model_rgb, base_model_mra,
        n_unfreeze=unfreeze_phase2,
        learning_rate=lr_phase2,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay
    )

    history_phase2 = model.fit(
        training_set,
        validation_data=validation_set,
        epochs=epochs_phase2,
        callbacks=callbacks,
        verbose=1
    )

    history["loss"].append(history_phase2.history["loss"])
    history["val_loss"].append(history_phase2.history["val_loss"])
    history["accuracy"].append(history_phase2.history["accuracy"])
    history["val_accuracy"].append(history_phase2.history["val_accuracy"])
    print("Fase 2 completata.")

    # ==================== FASE 3: Fine-tuning esteso ====================
    print("\n" + "=" * 70)
    print(f"FASE 3: Fine-tuning (ultimi {unfreeze_phase3} layer per backbone)")
    print("=" * 70)

    model = unfreeze_model(
        model, base_model_rgb, base_model_mra,
        n_unfreeze=unfreeze_phase3,
        learning_rate=lr_phase3,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay
    )

    history_phase3 = model.fit(
        training_set,
        validation_data=validation_set,
        epochs=epochs_phase3,
        callbacks=callbacks,
        verbose=1
    )

    history["loss"].append(history_phase3.history["loss"])
    history["val_loss"].append(history_phase3.history["val_loss"])
    history["accuracy"].append(history_phase3.history["accuracy"])
    history["val_accuracy"].append(history_phase3.history["val_accuracy"])
    print("Fase 3 completata.")

    # ==================== FASE 4: Fine-tuning completo ====================
    print("\n" + "=" * 70)
    print("FASE 4: Fine-tuning completo (tutti i layer)")
    print("=" * 70)

    model = unfreeze_model(
        model, base_model_rgb, base_model_mra,
        n_unfreeze=len(base_model_rgb.layers),
        learning_rate=lr_phase4,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay
    )

    history_phase4 = model.fit(
        training_set,
        validation_data=validation_set,
        epochs=epochs_phase4,
        callbacks=callbacks,
        verbose=1
    )

    history["loss"].append(history_phase4.history["loss"])
    history["val_loss"].append(history_phase4.history["val_loss"])
    history["accuracy"].append(history_phase4.history["accuracy"])
    history["val_accuracy"].append(history_phase4.history["val_accuracy"])

    print("\n" + "=" * 70)
    print("TRAINING COMPLETATO!")
    print("=" * 70)

    return model, history


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    # Training
    model, history = train_model()

    # Salvataggio modello e history
    model_name = "CrossAttn_IF"
    model.save(os.path.join(result_folder, f"model_{model_name}.keras"))
    np.save(os.path.join(result_folder, f"history_{model_name}.npy"), history)
    print(f"\nModello salvato: model_{model_name}.keras")

    # Plot training history
    plot_training_history(history, result_folder, model_name)

    # Valutazione
    metrics = evaluate_model(model, X_test_rgb, X_test_mra, y_test, result_folder, model_name)

    print("\n" + "=" * 70)
    print("TUTTI I RISULTATI SALVATI!")
    print("=" * 70)
    print(f"Cartella output: {result_folder}")
