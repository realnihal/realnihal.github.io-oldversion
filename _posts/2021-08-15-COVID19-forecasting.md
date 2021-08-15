---
layout: post
title: "Forecasting COVID-19 Case in India with deep learning"
subtitle: "Using deep learning to try and predict the near future in Indian Covid cases."
background: '/img/posts/covid19/covid19.jpeg'
---
## Introduction

The goal of this project is to give a fair estimate of covid cases in India. I came across an published article on [Forecasting COVID-19 cases](https://reader.elsevier.com/reader/sd/pii/S2211379721000048?token=96B6C9E2813943F5D2FE4882F66A79AFA5E8779BC525996AA7E6F9EE1B924E254C50FC4994A800B07CE92EADF065D17B&originRegion=eu-west-1&originCreation=20210815022455). They were able to make predections with an error of less that 2%. Here I have tried to implement their learnings and try to make predictions for the next few days.

You can check out the complete code [here](https://github.com/realnihal/Forecasting-COVID-19-cases).

## Importing data

I have imported the covid-19 data from [this source](https://documenter.getpostman.com/view/10724784/SzYXXKmA). The data is collected by volunteers and pre-cleaned. We get access to various metrics but only interested in the "Daily Case" counts.

```python
# importing the Covid-19 time series data
import urllib.request
url = 'https://api.covid19india.org/csv/latest/case_time_series.csv'
filename = 'case_time_series.csv'
urllib.request.urlretrieve(url, filename)
```
We extract the required data from our dataset and plot it to visualize the initial conditions.

```python
# plotting the data
import matplotlib.pyplot as plt
daily_cases.plot(figsize=(12, 5))
plt.ylabel("Covid Casesy")
plt.title("Covid Cases per day in India", fontsize=16)
plt.legend(fontsize=14);
plt.show()
```


    
![png](\img\posts\covid19\output_4_0.png)
    

We have to normalize the data as to increase the accuracy of the model.

```python
# Normalizing the data
timesteps = daily_cases.index.to_numpy()
cases = daily_cases["cases"].to_numpy()
cases = cases/414280
```

The time-series data that we have must be converted into windows. Basically it defines the no of days the model looks into the past in-order to predict the future. I have chosen to have the window size as 30 and the predicting horizon of 1 day. You can check out the [full code](https://github.com/realnihal/Forecasting-COVID-19-cases) to get an idea of how I did it.
Training and testing data is created by spliting the windowed data that we have. I have used a spilt ratio of 0.2.
We are creating a model checkpointing callback using the [tensorflow callback function](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint). This allows us to save only the best model that is trained across many epochs.


```python
import tensorflow as tf
# Let's build an Stacked LSTM model with the Functional API

inputs = layers.Input(shape=(WINDOW_SIZE))
x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs) # expand input dimension to be compatible with LSTM
x = tf.keras.layers.layers.LSTM(128, activation="relu", return_sequences=True)(x) 
x = tf.keras.layers.layers.Dropout(0.2)(x)
x = tf.keras.layers.layers.LSTM(128, activation="relu")(x)
x = tf.keras.layers.layers.Dropout(0.2)(x)
x = tf.keras.layers.layers.Dense(32, activation="relu")(x)
output = tf.keras.layers.layers.Dense(HORIZON)(x)
model_5 = tf.keras.Model(inputs=inputs, outputs=output, name="model_5_lstm")

# Compile model
model_5.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(0.0005))

history = model_5.fit(train_windows,
            train_labels,
            epochs=150,
            verbose=0,
            batch_size=128,
            validation_data=(test_windows, test_labels),
            callbacks=[create_model_checkpoint(model_name=model_5.name)])
```
the main reason for stacking LSTM is to allow for greater model complexity. In case of a simple feedforward net we stack layers to create a hierarchical feature representation of the input data to then use for some machine learning task. The same applies for stacked LSTM's.
At every time step an LSTM, besides the recurrent input. If the input is already the result from an LSTM layer (or a feedforward layer) then the current LSTM can create a more complex feature representation of the current input. This model and its parameter were derieved from the extensive testing done in this [paper](https://reader.elsevier.com/reader/sd/pii/S2211379721000048?token=96B6C9E2813943F5D2FE4882F66A79AFA5E8779BC525996AA7E6F9EE1B924E254C50FC4994A800B07CE92EADF065D17B&originRegion=eu-west-1&originCreation=20210815022455).

```python
# evaluating the best model
model_5 = tf.keras.models.load_model("model_experiments/model_5_lstm/")
model_5.evaluate(test_windows, test_labels)
```
 0.027053499594330788



## Results

we can see that we have achieved an error of 2.7% which is slightly higher than the original [paper](https://reader.elsevier.com/reader/sd/pii/S2211379721000048?token=96B6C9E2813943F5D2FE4882F66A79AFA5E8779BC525996AA7E6F9EE1B924E254C50FC4994A800B07CE92EADF065D17B&originRegion=eu-west-1&originCreation=20210815022455). Lets try to use this model to predict the future cases.


```python
def make_preds(pcases, model):
    no_of_preds = 100
    for i in range(no_of_preds):
        eval_case = pcases[-30:].reshape(1,30)
        pred = model.predict(eval_case)
        pcases = np.append(pcases,pred)
    return pcases
pred_cases = make_preds(cases, model_5)
pred_cases[-10:]
```




    array([0.21659741, 0.20328695, 0.19203615, 0.18219881, 0.17360687,
           0.16575938, 0.1582801 , 0.15125737, 0.14482188, 0.13891993])




```python
plt.figure(figsize=(12,5))
plt.plot(pred_cases)
plt.show()
```


    
![png](\img\posts\covid19\output_17_0.png)
    


We can see the rising trend of an upcoming third wave in the country. We have to consider a lot of things before we take this model seriously, such as:

1. We are using a single feature (univariate) to make the prediction. this may not be accurate as the real trends could be more correlated to  other factors.

2. The further in the future we want to predict, the less accurate the model becomes. This means that the actual slope may not be accurate. The peak or duration of the third wave might by varying a lot.

But nevertheless, this is an alarming sign that the public should be prepared. I really wish that this does'nt happen and the model is wrong, but it's still a good idea to increase precautions and try to save yoursleves.

[Click here to check out the complete code](https://github.com/realnihal/Forecasting-COVID-19-cases).

Please feel free to share your thoughts on this at any of my socials (*linked below*). Would love to hear from you. Be safe and Peace out!
