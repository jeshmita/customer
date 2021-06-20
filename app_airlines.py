# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 17:17:02 2021

@author: jeshm
"""

import numpy as np
# 1. Library imports
import uvicorn
from fastapi import FastAPI
from Airline_Satisfaction import AirlineSatisfaction
import pickle

# Create the app object
app = FastAPI()
pickle_in = open('XGB1.pkl',"rb")
xGB=pickle.load(pickle_in)
variety_mappings = {0: 'Not Satisfied or Neutral', 1: 'Satisfied'}


# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# Route with a single parameter, returns the parameter within a message
# Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome ': f'{name}'}

# Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_satisfaction(data:AirlineSatisfaction):
    data = data.dict()
    age = data['age']
    inflight_wifi_service = data['inflight_wifi_service'] 
    departure_arrival_time_convenient = data['departure_arrival_time_convenient']
    ease_of_online_booking = data['ease_of_online_booking']
    gate_location = data['gate_location']
    food_and_drink = data['food_and_drink']
    online_boarding = data['online_boarding']
    seat_comfort = data['seat_comfort']
    inflight_entertainment = data['inflight_entertainment']
    onboard_service = data['onboard_service']
    leg_room_service = data['leg_room_service']
    baggage_handling = data['baggage_handling']
    checkin_service = data['checkin_service']
    inflight_service = data['inflight_service']
    cleanliness = data['cleanliness']
    departure_delay_in_minutes = data['departure_delay_in_minutes']
    arrival_delay_in_minutes= data['arrival_delay_in_minutes']
    gender_Male = data['gender_Male']
    customer_type_disloyalCustomer = data['customer_type_disloyalCustomer']
    type_of_travel_PersonalTravel  = data['type_of_travel_PersonalTravel']	
    customer_class_Eco  =data['customer_class_Eco']	
    customer_class_EcoPlus =data['customer_class_EcoPlus']
    features = np.array([age, inflight_wifi_service, departure_arrival_time_convenient, ease_of_online_booking, gate_location, food_and_drink, online_boarding, seat_comfort, inflight_entertainment, onboard_service, leg_room_service, baggage_handling, checkin_service, inflight_service, cleanliness, departure_delay_in_minutes, arrival_delay_in_minutes, gender_Male, customer_type_disloyalCustomer, type_of_travel_PersonalTravel, customer_class_Eco, customer_class_EcoPlus])
    test = features.reshape(1,-1)
    prediction = variety_mappings[xGB.predict(test)[0]]
    return {'prediction': prediction}

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
