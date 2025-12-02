# IU-CSProject-Navjot-Kaur

# Predictive Train Delay and Disruption Analysis System (Deutsche Bahn)

# Problem Statement:
Passengers on Deutsche Bahn often face irregular train delays and disruptions because of variety of factors such as weather conditions,track maintenance,platform congestion ,vehicle change and peak hour traffic.Most of the information provided by the available systems is reactive and discriptive. It means that passengers are informed about the delays afterwords rather than before delays happen.This kind of reactive information or strategy effects planning of both passengers and train operators and that result in irritation,missing connection and poor resource management.DEcision makers find it difficult to take measures to prevent them because current tools are not providing precise information abbout why disruptions occur.

# Goal:
- **Predict train delays and disruptions** before they occur using historical and real-time data.  
- **Provide accurate projections** of delay times across various routes and time periods.  
- **Identify main causes** of delays such as bad weather, track maintenance, and traffic congestion.  
- **Use predictive data** to help train operators make better operational choices.  
- **Enhance passenger experience** by delivering reliable travel information and early alerts about delays.  
- **Offer a simple and clear display** for showing train status and prediction results.  
- **Develop a modular AI system** that allows continuous learning, updates, and scalability.  


# Tech Stack:
| Component            | Technology / Library                        | Description                                                                 |
|---------------------|--------------------------------------------|-----------------------------------------------------------------------------|
| **Frontend**         | React.js, Plotly.js                         | User interface for visualizing real-time predictions and analytics.        |
| **Backend API**      | FastAPI, Flask                              | REST APIs for serving predictions, explanations, and handling requests.    |
| **Database**         | PostgreSQL                                  | Store structured data, historical data, and prediction results.            |
| **AI Engine**        | XGBoost, SHAP                | Machine learning models and interpretability for predicting delays.        |
| **Data Ingestion**   | Python, Airflow, APIs                        | Collection, cleaning, and preprocessing of live and historical data.       |
| **Integration**      | REST APIs, WebSockets                        | Connecting frontend, backend, and data pipelines for real-time updates.    |
| **Data Sources**     | OpenWeatherMap API, Calendar/Holiday API | Source data for train schedules, weather conditions, and holidays.        |

# How to Run Project
## Backend
1. install Python dependencies
pip install -r requirements.txt
2. Run the backend server
python main.py

## Frontend
1. Install Node.js dependencies
npm install
2. Start the development server
npm run dev


# Project Risk
1. Description: The Predictive Train Delay and Disruption Analysis system needs good, reliable data to work well. However Deutsche Bahn’s data like live delay reports, schedules, and disruption logs might not always be complete or easy to use. Other data sources, like weather updates or calendar events, can also have missing information, slow updates, or access limits. If the data is poor or limited, the machine learning model won’t perform as well.
 
2. Impact: If the data is inaccurate, incomplete, or hard to get, the system might give wrong delay predictions. This would lower the models accuracy and make it less useful for passengers and operators. Overall poor data could prevent the project from achieving its goal of providing reliable, real time train information.
   
4. Mitigation: To handle this risk, the project will combine open Deutsche Bahn data with live weather info from OpenWeatherMap and calendar data from Google Calendar. The data will be cleaned and corrected to fix missing or inconsistent values. When data is incomplete, historical or simulated data can be used for training the model. The system will also retrain and validate the model regularly to keep predictions accurate over time.

# Phase Status
1. Conception Phase - Done
2. Development Phase - Done
3. Finalisation Phase - Done
