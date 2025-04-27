from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam

# Building a stronger U-Net
inputs = Input((256, 256, 3))

# Encoder
c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
p2 = MaxPooling2D((2, 2))(c2)

# Bottleneck
c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

# Decoder
u1 = UpSampling2D((2, 2))(c3)
u1 = concatenate([u1, c2])
c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

u2 = UpSampling2D((2, 2))(c4)
u2 = concatenate([u2, c1])
c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# --------- Train the model here ---------
# X_train : your input images (shape: N, 256, 256, 3)
# y_train : your ground-truth masks (shape: N, 256, 256, 1)

# Example:
# model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1)

# --------- After training, save it ---------
model.save('unet_model.h5')
print("Model saved as unet_model.h5")
