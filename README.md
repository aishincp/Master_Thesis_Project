## Master_Thesis_Project
# Dynamic Smart Office Environment for Occupancy Detection Utilizing Neural Network

### 1. Introduction and Initial Idea
In today's modern workplaces, optimizing space utilization, ensuring privacy, and improving safety are crucial challenges. The increasing demand for smart office environments has driven the need for advanced technologies that can efficiently and accurately determine whether a space is occupied or unoccupied. My thesis project set out to address these challenges by developing a dynamic occupancy detection system using a Long Short-Term Memory (LSTM) neural network model. The primary objective was to create a system that is both reliable and privacy-preserving, while also specifically leveraging the ultrasonic sensor integrated with the Red Pitaya device. This integration was crucial because the ultimate goal was to ensure that in future this LSTM model could be seamlessly incorporated into the Red Pitaya software system, enabling the Red Pitaya device to be deployed in smart office spaces for real-time occupancy detection.

##### *Initial Considerations:*
When conceptualizing this project, it became clear that traditional occupancy detection systems - often relying on simple motion sensors or cameras - have significant limitations. These systems frequently struggle with issues like false positives or negatives, and camera-based solutions raise concerns about surveillance and data privacy. Recognizing these challenges, exploring more sophisticated approaches that could offer better accuracy and privacy protection, while being compatible with the Red Pitaya platform.
Choice of Technology:
Given the sequential nature of data - where sensor readings need to be analyzed over time - a Recurrent Neural Network (RNN) was identified as a natural fit for this work. However, RNNs are known to suffer from issues like vanishing and exploding gradients, particularly with long sequences. To overcome these challenges, through a rigorous research Long Short-Term Memory (LSTM) network, an advanced variant of RNNs was chosen. LSTMs effectively capture long-term dependencies in time-series data, particularly well-suited for analyzing the complex patterns in occupancy data.

### 2. Research Work
The core of this research focused on creating a highly accurate and privacy-preserving occupancy detection system for smart office environments. The key research question guiding this work was: How to design a highly reliable and privacy-preserving occupancy detection system for a dynamic smart office environment, using ultrasonic sensor integrated with Red Pitaya controller device?

### Scope of Research:
To address this question, the research focused on integrating labeled sensor data from an ultrasonic sensor integrated with the Red Pitaya controller device, leveraging deep learning techniques. The goal was to ensure  that the system could seamlessly adapt to the dynamic nature of the smart workplaces while preserving occupant privacy and ensuring that the Red Pitaya device could be deployed effectively in such environments.


### Research Methodology:

##### *Physical Environment Setup*:

The physical setup involved placing the Red Pitaya equipped ultrasonic sensor in a simulated office space, collecting time-series data on occupant movements. To ensure high data accuracy, to be named as "Auxiliary System" (including the same above-mentioned integrated Ultrasonic sensor with Red Pitaya device along with developed YOLO object detection model) to label the sensor data, classifying it as 'occupant' and 'non-ocuupant'. This labeled data was then used to train an LSTM network, which learned to detect patterns in the data with high accuracy.

![image](https://github.com/user-attachments/assets/0f2cf96e-7c9e-4866-b3d7-18fdd2547353)

    Fig: Occupant Detection Environment Setup

![image](https://github.com/user-attachments/assets/d73e49d6-06ae-4db6-8302-b97ba5a56a86)

    Fig: Non-Occupant Detection Environment Setup 

##### *Data Collection & Label Extraction:*

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

##### *Developing a Highly Reliable Deep Learning Model (RNN-LSTM):*
Model Development: Developed an RNN model using LSTM networks to process FFT data and capture temporal dependencies in occupancy patterns.
Model Integration: Designed the model for seamless future integration into Red Pitaya's software ecosystem for real-time occupancy detection.

##### *Ensuring Privacy:*
Privacy Focus: Ensured privacy by training the model solely on FFT data, derived from sound signals, while using image data only for label extraction. This approach maintains high accuracy without compromising individual privacy.

### 5. Why LSTM for Occupancy Detection?

![LSTM1](https://github.com/user-attachments/assets/2265e4ac-6840-4705-9df4-eff2663a5a50)

Fig: LSTM Neural Network Layout [[Source]](https://medium.com/@pradnyakokil24/fruit-and-vegetable-identification-system-using-efficient-convolutional-neural-networks-for-146f1fe7c139)

##### *Sequential Data Handling:*
LSTM networks are particularly well-suited for this project because they are designed to handle sequential data, making them suitable for time-series analysis, such as continuous stream of sensor readings in occupancy detection.

##### *Handling Long-Term dependencies:*
LSTMs overcome the limitations of traditional RNNs by using a gating mechanism, which allows them to retain information over long sequences. This is critical in occupancy detection, where the system needs to understand context over time (e.g. distinguishing between brief movements and sustained presence).

### 6. Experimentation with LSTM Architectures
##### *Initial Experiments:*
A series of experiments were conducted to determine the optimal LSTM architecture for the occupancy detection model. These experiments varied -the number of hidden layers (LSTM layers) and their configurations to identify the best-performing model in terms of accuracy, training time, and computational efficiency. 

__Experiment 1:__
- Configuration: 2 LSTM layers (64, 64 units)
- Optimizer: Adam with lr = 0.0001

| Parameters | Values | 
| --- | --- |
| Test Accuracy | 97.54% |
| Test Loss | 0.0795 |
| F1 Score | 0.9772 |
| Training Time | ~3 hours |

__Observation__: This configuration achieved good accuracy but showed potential for improvement in loss and precision.

__Experiment 2:__
- Configuration: 3 LSTM layers (64, 64, 64 units)
- Optimizer: Adam with lr = 0.001

| Parameters | Values | 
| --- | --- |
| Test Accuracy | 98.08% |
| Test Loss | 0.0650 |
| F1 Score | 0.9822 |
| Training Time | ~5.6 hours |

__Observation__: This experiment yielded the best balance between accuracy, loss, and training time, indicating that three layers of 64 units each were optimal for this task.

__Experiment 3:__
- Configuration: 4 LSTM layers (128, 64, 64, 64 units)
- Optimizer: Adam with lr = 0.001

| Parameters | Values | 
| --- | --- |
| Test Accuracy | 97.65% |
| Test Loss | 0.0755 |
| F1 Score | 0.9781 |
| Training Time | ~6.7 hours |

__Observation__: While the model was still accurate, it did not significantly outperform the three-layer model and required more training time, making it less efficient.


### 6. Decision to Use 3 LSTM Layers of 64 Units

##### *Performance Optimization:*

Based on the experimental results, the three-layer LSTM model with 64 units in each layer was selected. This configuration provided the best balance between accuracy, computational efficiency, and model complexity. The model achieved a high test accuracy of 98.08% with minimal loss and a strong F1 score, which are critical for ensuring reliable occupancy detection.

##### *Choice of Adam Optimizer:*
The Adam optimizer was chosen for its adaptive learning rate and efficiency in handling sparse gradients, which are common in occupancy detection tasks where the difference between occupied and unoccupied states can be subtle. The learning rate of 0.001 was found to be optimal during your experiments, providing a good trade-off between convergence speed and model performance.
