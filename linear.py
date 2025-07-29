import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential 
from keras.layers import Dense

x = np.linspace(1, 1, 100)
y = 2*x + 20 +np.random.randn(100)
print(x)
print(y)

x.reshape(-1,1) # Error 1: This line doesn't modify x in place
print(x.shape)  # Error 2: x.shape will still be (100,)

model = Sequential() # Error 3: Incorrect way to instantiate Sequential model
model.add(Dense(1, input_dim=1, activation='linear')) # Error 4: 'model' object not defined yet
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x,y,epochs=100)

pred = model.predict(x)
plt.scatter(x,y,label = 'original data')
plt.plot(x, pred,label='predicted line')
plt.show() # Error 5: Trailing dot