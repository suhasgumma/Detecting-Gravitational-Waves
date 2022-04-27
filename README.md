# Problem Definition
Understanding Gravitational waves is the key to further probe into the workings of our Universe. GW signals are unimaginably tiny ripples in the fabric of space-time and even though the global network of GW detectors are some of the most sensitive instruments on the planet, the signals are buried in detector noise. Developing proper ML models to detect the Gravitational waves is the problem we are going to tackle.


# Challenges
* Processing the Signal.
* The time series should be transformed into other forms to be used with different ML models.  
* Dealing with large data. 
* High computational power is required especially when Deep Learning models come into the picture.


Deliverables
● As mentioned above, the objective of our project/model is to predict the presence of GW signals in the data. Our model will output a probability score(between 0 and 1) of the presence of GW Signal. 
The model developed will detect the signal with an AUC score of at least 0.75.

 
# Description of the Dataset
* The training set of time series data contains simulated gravitational wave measurements from a network of 3 gravitational wave interferometers (LIGO Hanford, LIGO Livingston, and Virgo). Each time series contains either detector noise or detector noise plus a simulated gravitational wave signal. 
 
* Each data sample (npy file) contains 3-time series (1 for each detector) and each spans 2 sec and is sampled at 2,048 Hz. 
 
* In summary, there are 560K npy files for training and 226K npy files for testing in which each npy file contains 3-time series data.
 
* And 2 CSV files. One for training and one for testing. Each contains 2 columns. The first column represents the npy file id(unique of course) and the second column contains the target (0 if GW wave is not present else 1).


# For more info, refer to the report uploaded.
