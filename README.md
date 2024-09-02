## Master_Thesis_Project
# Dynamic Smart Office Environment for Occupancy Detection Utilizing Neural Network

### 1. Introduction and Initial Idea
In today's modern workplaces, optimizing space utilization, ensuring privacy, and improving safety are crucial challenges. The increasing demand for smart office environments has driven the need for advanced technologies that can efficiently and accurately determine whether a space is occupied or unoccupied. My thesis project set out to address these challenges by developing a dynamic occupancy detection system using a Long Short-Term Memory (LSTM) neural network model. The primary objective was to create a system that is both reliable and privacy-preserving, while also specifically leveraging the ultrasonic sensor integrated with the Red Pitaya device. This integration was crucial because the ultimate goal was to ensure that in future this LSTM model could be seamlessly incorporated into the Red Pitaya software system, enabling the Red Pitaya device to be deployed in smart office spaces for real-time occupancy detection.

#### *Initial Considerations:*
When conceptualizing this project, it became clear that traditional occupancy detection systems - often relying on simple motion sensors or cameras - have significant limitations. These systems frequently struggle with issues like false positives or negatives, and camera-based solutions raise concerns about surveillance and data privacy. Recognizing these challenges, exploring more sophisticated approaches that could offer better accuracy and privacy protection, while being compatible with the Red Pitaya platform.
Choice of Technology:
Given the sequential nature of data - where sensor readings need to be analyzed over time - a Recurrent Neural Network (RNN) was identified as a natural fit for this work. However, RNNs are known to suffer from issues like vanishing and exploding gradients, particularly with long sequences. To overcome these challenges, through a rigorous research Long Short-Term Memory (LSTM) network, an advanced variant of RNNs was chosen. LSTMs effectively capture long-term dependencies in time-series data, particularly well-suited for analyzing the complex patterns in occupancy data.

### 2. Research Work
The core of this research focused on creating a highly accurate and privacy-preserving occupancy detection system for smart office environments. The key research question guiding this work was: **How to design a highly reliable and privacy-preserving occupancy detection system for a dynamic smart office environment, using ultrasonic sensor integrated with Red Pitaya controller device?**

### 3. Scope of Research
To address this question, the research focused on integrating labeled sensor data from an ultrasonic sensor integrated with the Red Pitaya controller device, leveraging deep learning techniques. The goal was to ensure  that the system could seamlessly adapt to the dynamic nature of the smart workplaces while preserving occupant privacy and ensuring that the Red Pitaya device could be deployed effectively in such environments.

![Proposed Model System Design](https://github.com/user-attachments/assets/438402f9-878f-4dc4-857a-6421ef02c54e)

### 4. Research Methodology
#### *Physical Environment Setup*:

The physical setup involved placing the Red Pitaya equipped ultrasonic sensor in a simulated office space, collecting time-series data on occupant movements. To ensure high data accuracy, to be named as "Auxiliary System" (including the same above-mentioned integrated Ultrasonic sensor with Red Pitaya device along with developed YOLO object detection model) to label the sensor data, classifying it as 'occupant' and 'non-ocuupant'. This labeled data was then used to train an LSTM network, which learned to detect patterns in the data with high accuracy.

![image](https://github.com/user-attachments/assets/0f2cf96e-7c9e-4866-b3d7-18fdd2547353)

    Fig: Occupant Detection Environment Setup

![image](https://github.com/user-attachments/assets/d73e49d6-06ae-4db6-8302-b97ba5a56a86)

    Fig: Non-Occupant Detection Environment Setup 

#### *Data Collection & Label Extraction:*

*Data Acquisition:*
Data collection is a crucial phase, a comprehensive dataset was collected utilizing an auxiliary system comprising an ultrasonic sensor integrated with Red Pitaya controller device, supplemented by a webcam-based object detection (YOLO) model to label the data. Over 130,000 labeled samples were collected, categorized into "Occupant Detected" and "Non-Occupant Detected". The data was saved in FFT format, was then labeled and merged into a comprehensive CSV file for model training. 

*Data Processing:*
The raw ultrasonic data was converted into Fast Fourier Transform (FFT) device format to capture frequency-domain features, which are particularly useful for detecting subtle patterns in occupancy data. These FFT data were merged with the labels derived from the YOLO model, creating a comprehensive dataset in NumPy format, later the labels were extracted from the image folders and were merged with the FFT data and all these information were collected and saved in CSV file. This dataset served as the foundation for training the LSTM model.

Here's a snippet of the code used for data processing:
```
# Function to extract labels from ultrasonic data filenames
def extract_label_ultrasonic(filename):
    # map "p" as Occupied/Person
    label_map = {"p": "1", "o": "0", "c": "0"}  # mapping both "o" and "c" as Unoccupied
    label_key = os.path.splitext(os.path.basename(filename))[0].split("_")[2].lower()
    return label_map.get(label_key, "unknown")

# Convert ultrasonic data from .npy files to CSV format and merge each sample with labels
def convert_ultrasonic_to_csv(ultrasonic_sensor_folder, output_csv):
    dataframes = []
    for root, dirs, files in os.walk(ultrasonic_sensor_folder):
        for file in files:
            if file.endswith(".npy"):
                filepath = os.path.join(root, file)
                data = np.load(filepath)
                ultrasonic_data = [i[0] for i in struct.iter_unpack('@h', data.tobytes()[64:])]
                tmp_df = pd.DataFrame([ultrasonic_data])
                tmp_df.insert(0, "label", [extract_label_ultrasonic(file)])
                dataframes.append(tmp_df)
    final_dataframe = pd.concat(dataframes, ignore_index=True)
    final_dataframe.to_csv(output_csv, index=False)
```

#### Data Loading:
```
# Loading the input filepath
fft_data = pd.read_csv(r'C:\Users\yousuf\source\repos\IndividualProject_Mr.PechThesisDataset&Model\ThesisModel\Thesis_Datasets(WithLabels)\ultrasonic_sensor_data\FFT_Labeled_Data_70K.csv')
fft_data.shape
```

#### Data Preprocessing and Feature Extraction:

```
# Extracting features (excluding the first column which has the labels)
X_fft = fft_data.iloc[:, 1:].values

# Extracting labels (using the first column as the labels)
y_fft = fft_data.iloc[:, 0].values.astype(int)

# Display the shapes of features and labels to verify the extraction
print("\nShape of features (X_fft):", X_fft.shape)
print("Shape of labels (y_fft):", y_fft.shape)
```
     Shape of features (X_fft): (136425, 85)
     Shape of labels (y_fft): (136425,)
     
```
# Splitting training & testing set parameters
X_fft_train, X_fft_test, y_fft_train, y_fft_test = train_test_split(X_fft, y_fft, test_size=0.3, random_state=42)
```

     Shapes of input data:
     X_fft_train shape: (95497, 85)
     X_fft_test shape: (40928, 85)

     Shapes of labels:
     y_fft_train shape: (95497,)
     y_fft_test shape: (40928,)

     Distribution of labels in the training set:
     1    51491
     0    44006
     Name: count, dtype: int64

     Distribution of labels in the testing set:
     1    21908
     0    19020
     Name: count, dtype: int64

### 5. Developing a Highly Reliable Deep Learning Model (RNN-LSTM)

__Model Development:__ Developed an RNN model using LSTM networks to process FFT data and capture temporal dependencies in occupancy patterns.
__Model Integration:__ Designed the model for seamless future integration into Red Pitaya's software ecosystem for real-time occupancy detection.

#### *Ensuring Privacy:*
__Privacy Focus:__ Ensured privacy by training the model solely on FFT data, derived from sound signals, while using image data only for label extraction. This approach maintains high accuracy without compromising individual privacy.
### 6. Why LSTM for Occupancy Detection?

![LSTM1](https://github.com/user-attachments/assets/2265e4ac-6840-4705-9df4-eff2663a5a50)

Fig: LSTM Neural Network Layout [[Source]](https://medium.com/@pradnyakokil24/fruit-and-vegetable-identification-system-using-efficient-convolutional-neural-networks-for-146f1fe7c139)

#### *Sequential Data Handling:*
LSTM networks are particularly well-suited for this project because they are designed to handle sequential data, making them suitable for time-series analysis, such as continuous stream of sensor readings in occupancy detection.

#### *Handling Long-Term dependencies:*
LSTMs overcome the limitations of traditional RNNs by using a gating mechanism, which allows them to retain information over long sequences. This is critical in occupancy detection, where the system needs to understand context over time (e.g. distinguishing between brief movements and sustained presence).

### 7. Implementation and Integration
#### *Model Implementation:*

The LSTM model was implemented in Python using the Keras library with TensorFlow as the backend. The architecture was designed to include dropout layers after each LSTM layer to prevent over-fitting, which is crucial in maintaining the model's ability. The model architecture includes:

- __Input LSTM Layer__: Accepts preprocessed sequential data
- __Two LSTM Layers__: The first layer with 64 units & the second layer again with 64 units to effectively learn complex temporal patterns
- __Dropout Layer__: Applied after each LSTM layers with a rate of 0.5 to prevent overfitting
- __Output (Dense) Layer__: A single neuron with sigmoid activation for binary classification (occupied vs. non-occupied)

#### *Hyperparameter Selection*:
Hyperparameters were meticulously tuned to optimize model performance:

- __Learning Rate__: Set to 0.001 based on experimentation for optimal convergence speed.
- __Batch Size__: Chosen as 32 to balance training speed and stability.
- __Epochs__: The model was trained over 50 epochs, with early stopping implemented to prevent overfitting.
- __K-Fold Cross-Validation__: Ensured model robustness by validating across multiple data splits.
  
Here's a simplified version of the core model structure:

```
# Building LSTM layers
rnn_model = Sequential()

# 1st LSTM input hidden layer with 64 units 
rnn_model.add(LSTM(64, input_shape=(X_fft_train.shape[1], 1), return_sequences=True))
rnn_model.add(Dropout(0.5))

# 2nd LSTM hidden layer with 64 units
rnn_model.add(LSTM(64, return_sequences=True))
rnn_model.add(Dropout(0.5))

# 3rd LSTM hidden layer with 64 units
rnn_model.add(LSTM(64))
rnn_model.add(Dropout(0.5))

# Output layer with Sigmoid activation for binary classification
rnn_model.add(Dense(1, activation='sigmoid'))

# Model compilation with Adam optimizer
rnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Model training
# Train the LSTM model and store the training history
history = rnn_lstm_model.fit(X_fft_train.reshape(X_fft_train.shape[0], X_fft_train.shape[1], 1), y_fft_train, 
                             epochs=50, batch_size=32, validation_data=(X_fft_test.reshape(X_fft_test.shape[0], 
                            X_fft_test.shape[1], 1), y_fft_test), callbacks=[early_stopping, checkpoint])
```
The whole code is avialable here : [__See Here__](https://github.com/aishincp/Master_Thesis_Project/blob/main/Code/Thesis_OccupancyDetection_LSTM_Model.ipynb)

#### *Model Selection Experiments:*
A series of experiments were conducted to determine the optimal LSTM architecture for the occupancy detection model. These experiments varied -the number of hidden layers (LSTM layers) and their configurations to identify the best-performing model in terms of accuracy, training time, and computational efficiency. 

__Experiment 1: Initial Model Selection [LSTM vs. GRU vs. BiLSTM]__:

The occupancy detection model developed during this research goes through rigorous testing and evaluation to ensure its accuracy and reliability. Various experiments were conducted using different Recurrent Neural Network (RNN) architectures, including LSTM, GRU, and BiLSTM, to determine the most effective model for this application. For a fair comparison, the same hyperparameters for all models were set, as can be seen below. 

- Number of Layers: 3 layers
- Units per Layer: 64 units
- Learning rate: 0.001
- Dropout Rate: 0.5

| NN Model | Accuracy | ROC AUC | Loss | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM | 0.9817 | 0.9814 | 0.0630 | 0.9802 | 0.9858 | 0.9830 |
| GRU | 0.9726 | 0.9721 | 0.0872 | 0.9701 | 0.9790 | 0.9745 |
| BiLSTM | 0.9768 | 0.9764 | 0.0765 | 0.9758 | 0.9810 | 0.9784 |

__Observation__: The results summarized in Table 4.6 below showed that the LSTM model outperformed GRU and BiLSTM across all metrics, particularly in accuracy and ROC AUC score. These metrics indicated that LSTM was more effective at capturing the temporal dependencies in the data, making it the preferred choice for further optimization.

__Experiment 2: 2.	Layer Configuration Optimization__:

After choosing the LSTM architecture, the next step was to examine the optimal number of LSTM layers. Four different configurations were tested to see how the depth of the network affected performance, and they are listed below:

- Configuration A: 4 LSTM layers with 128 units in the input layer, 64 units in the three hidden layers, and a Dense output layer
- Configuration B: 4 LSTM layers with 64 units in the input layer, 64 units in the three hidden layers, and a Dense output layer
- Configuration C: 3 LSTM layers with 64 units in the input layer, 64 units in the two hidden layers, and a Dense layer
- Configuration D: 2 LSTM layers with 64 units in the input layer, 64 units in the hidden layer, and a Dense output layer

| Configurations | Layers | Accuracy | ROC AUC | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- | --- |
| A | 128, 64, 64, 64, 1 | 0.9765 | 0.9762 | 0.9762 | 0.9799 | 0.9781 |
| B | 64, 64, 64, 64, 1 | 0.9735 | 0.9730 | 0.9705 | 0.9803 | 0.9754 |
| C | 64, 64, 64, 1 | 0.9817 | 0.9814 | 0.9802 | 0.9858 | 0.9830 |
| D | 64, 64, 1 | 0.9800 | 0.9797 | 0.9795 | 0.9769 | 0.9767 |

__Observation__: Configuration C (3 LSTM layers with 64 units each) achieved the highest accuracy and overall performance. This setup was found to be a good balance between model complexity and performance, as adding more layers did not significantly improve the results and could potentially lead to overfitting.

__In summary, through a systematic evaluation of different RNN architectures, layer configurations, and learning rates, the LSTM model with three layers and a learning rate of 0.001 was identified as the best-performing hyperparameter for the model of occupancy detection. This process ensured that the model was accurate and efficient, capable of capturing the necessary temporal dependencies in the data.__

### 8. Model Evaluation

__1. Confusion Matrix for Training & Testing Data:__

![image](https://github.com/user-attachments/assets/48385d97-72a6-4e3e-8db4-8d8b7fa60a7c)

     Fig: Confusion Matrix of Training Data

![image](https://github.com/user-attachments/assets/fa7ba14e-1a44-4c1a-b458-2c9b50556db8)

     Fig: Confusion Matrix of Testing Data

__2. Performance Metrics for Training & Testing Data:__

![image](https://github.com/user-attachments/assets/844932d0-9801-49a0-b205-5331a554893e)

     Fig: Training accuracy 

![image](https://github.com/user-attachments/assets/8e749a2d-0f93-4dc5-8757-dbb15dcc4f0f)

     Fig: Training loss 

__3. Saving the Model:__

The trained and tested occupancy detection model is saved in TensorFlow Keras format to ensure its effectiveness in real-time applications and integration into other software systems. This format preserves the model’s architecture, weights, and training setup, allowing for seamless future incorporation into the integrated system consisting of a Red Pitaya controller with an ultrasonic sensor.

```
# Load the model weights
rnn_lstm_model.load_weights('best_rnn_lstm_model.weights.h5')

# Save the model with weights in Keras TensorFlow format
tf.keras.models.save_model(rnn_lstm_model, 'rnn_lstm_model.keras')
```
__Saved Model__ [See Here](https://github.com/aishincp/Master_Thesis_Project/tree/main/Code)
