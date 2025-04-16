import os
import numpy as np
import pandas as pd
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

# ======== LOAD DATA ========
def load_data(input_size=(64, 64), data_path='D:/Traffic_Sign_Recognition/GTSRB/Final_Training/Images'):
    pixels, labels = [], []
    for dir_name in os.listdir(data_path):
        if not dir_name.isdigit(): continue
        class_dir = os.path.join(data_path, dir_name)
        csv_file = os.path.join(class_dir, f'GT-{dir_name}.csv')
        df = pd.read_csv(csv_file, sep=';')

        for _, row in df.iterrows():
            img_path = os.path.join(class_dir, row['Filename'])
            img = imageio.imread(img_path)
            roi = img[row['Roi.Y1']:row['Roi.Y2'], row['Roi.X1']:row['Roi.X2']]
            img_resized = cv2.resize(roi, input_size)
            pixels.append(img_resized)
            labels.append(row['ClassId'])

    return np.array(pixels), np.array(labels)

print("🚀 Đang load dữ liệu...")
X, y = load_data()
X = X / 255.0
y = to_categorical(y)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ======== BUILD MODEL ========
def build_model(input_shape=(64, 64, 3), output_size=43):
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(16, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.2),

        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.2),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.2),

        Flatten(),
        Dense(2048, activation='relu'),
        Dropout(0.3),
        Dense(1024, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(output_size, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])
    return model

# ======== TRAINING ========
print("🏗️ Đang xây dựng và huấn luyện mô hình...")
model = build_model()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)
model.save("traffic_sign_model_tf4.h5")
print("✅ Mô hình đã được lưu thành công!")

# ======== ACCURACY/LOSS PLOT ========
print("📈 Vẽ biểu đồ độ chính xác và loss...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy per Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss per Epoch")
plt.legend()
plt.savefig("accuracy_loss.png")
plt.close()

# ======== TEST & CONFUSION MATRIX ========
print("🔎 Đang đánh giá mô hình trên tập test...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"🎯 Độ chính xác trên tập test: {test_acc:.4f}")

y_true = np.argmax(y_test, axis=1)
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")
plt.close()

# ======== CLASSIFICATION REPORT ========
report = classification_report(y_true, y_pred_classes, digits=4)
with open("classification_report.txt", "w") as f:
    f.write(report)
print("📄 classification_report.txt đã được tạo.")

report_dict = classification_report(y_true, y_pred_classes, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
df_report.to_csv("metrics_per_class.csv")
print("📊 metrics_per_class.csv đã được lưu.")

# ======== TOP NHẦM LẪN ========
flat_cm = cm.copy()
np.fill_diagonal(flat_cm, 0)
most_confused = np.unravel_index(np.argsort(flat_cm.ravel())[-5:], flat_cm.shape)
print("🤯 5 cặp lớp nhầm lẫn nhiều nhất:")
for i in range(5):
    actual, predicted = most_confused[0][i], most_confused[1][i]
    print(f"Lớp {actual} bị nhầm thành {predicted}: {cm[actual][predicted]} lần")

# ======== MISCLASSIFIED EXAMPLES ========
incorrect_indices = np.where(y_true != y_pred_classes)[0]
plt.figure(figsize=(12, 10))
for i, idx in enumerate(incorrect_indices[:16]):
    plt.subplot(4, 4, i+1)
    plt.imshow(X_test[idx])
    plt.title(f"T: {y_true[idx]} / P: {y_pred_classes[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig("misclassified_examples.png")
plt.close()
print("📷 misclassified_examples.png đã được tạo.")
