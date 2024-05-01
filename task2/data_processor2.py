import ipaddress
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import tensorflow as tf

df = pd.read_csv("Darknet.csv")
df = df.drop(["Flow ID", "Timestamp", "Label2","Src IP","Dst IP"], axis=1)
df = df.dropna()

label_encoder1 = LabelEncoder()
df['Label1'] = label_encoder1.fit_transform(df['Label1'])

df.to_csv("processed.csv", index =False)

df2 = pd.read_csv("processed.csv")
data = df2.drop(['Label1'],axis=1)
labels = df2['Label1']

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(250, activation='relu', input_shape=(features.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=512, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
with open('accuracy2.txt', 'w') as f:
    f.write(f'Test Accuracy: {acc:.4f}\n')
