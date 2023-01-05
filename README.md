# Event-camera-classification

Academic Kaggle Challenge : Classification of event-based sensors (also known as neuromorphic camera) data.\

- First, an event is 4-tuple $(x, y, p, t)$, where $(x, y)$ denotes the pixel position related to the event, $p$ is the polarity of the event (= 1 or 0 whether the brightness has increased or decreased) and $t$ is the timestamp : time (in $\mu s$ ) since the begining of the recording, at which the event has been detected.

An event file is a dataframe where rows are events\


The dataset used here consists of 320 event files.\

This dataset is provided by Prophesee Event camera for educational purposes. This type of sensor is also known in literature as Dynamic Vision Sensor, and has large application in computer vision, specially where refresh rate has to be high (of the order of s\micro $$) We have 10 classes, 32 examples for each class, and our task is to classify new examples.\

For this purpose, a preprocessing have been made in order to convert those event files into sequences of frames (conventional camera data), then a PCA have been performed to reduce dimensionality.
A random forest has been applied upon the result.\

Performances are detailed in the pdf, the true accuracy (on real data, in the Kaggle Challenge) is about 0.77
