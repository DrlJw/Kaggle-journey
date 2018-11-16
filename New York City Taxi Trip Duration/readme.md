# Introduction
In this competition, Kaggle is challenging you to build a model that predicts ***the total ride duration of taxi trips in New York City.***   
Your primary dataset is one released by the NYC Taxi and Limousine Commission,   
which includes ***pickup time, geo-coordinates, number of passengers, and several other variables***.  

# Evaluation 
The evaluation metric for this competition is **Root Mean Squared Logarithmic Error**.

The ***RMSLE*** is calculated as

![](http://latex.codecogs.com/gif.latex?\\epsilon=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(log(p_i+1)+log(a_i+1))^{2}})

Where:

ϵ is the RMSLE value (score)  
n is the total number of observations in the (public/private) data set,  
![](http://latex.codecogs.com/gif.latex?\$p_i$) is your prediction of trip duration, and  
![](http://latex.codecogs.com/gif.latex?\$a_i$) is the actual trip duration for i.   
log(x) is the natural logarithm of x  

# Data
* **id** - a unique identifier for each trip  
* **vendor_id** - a code indicating the provider associated with the trip record  
* **pickup_datetime** - date and time when the meter was engaged  
* **dropoff_datetime** - date and time when the meter was disengaged  
* **passenger_count** - the number of passengers in the vehicle (driver entered value)  
* **pickup_longitude** - the longitude where the meter was engaged  
* **pickup_latitude** - the latitude where the meter was engaged  
* **dropoff_longitude** - the longitude where the meter was disengaged  
* **dropoff_latitude** - the latitude where the meter was disengaged  
* **store_and_fwd_flag** - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor   
because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip  
* **trip_duration** - duration of the trip in seconds  
