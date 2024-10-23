from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import time
import csv
import requests
import pandas as pd
import matplotlib.pyplot as plt

application = Flask(__name__)
#predict_link = 'http://127.0.0.1:8080/predict'
predict_link = 'http://pra5-env.eba-epmsshkf.us-east-2.elasticbeanstalk.com//predict'

# Load model and vectorizer at the start to avoid reloading during each request
with open('basic_classifier.pkl', 'rb') as fid:
    loaded_model = pickle.load(fid)

with open('count_vectorizer.pkl', 'rb') as vd:
    vectorizer = pickle.load(vd)
    
 
text = ['COVID 19 saved lives', 'Justin Trudeau is the PM of Canada',
        'The earth is flat', 'UofT is the best university in Canada']
    
# Route for basic check
@application.route('/')
def index():
    return "Flask Message123!"

# Route for prediction
@application.route('/predict', methods=['GET','POST'])
def predict():
   
    # Use the loaded model to make predictions
    predictions=[]
    
    for i in text:
        predictions.append(loaded_model.predict(vectorizer.transform([i]))[0])
    
    return jsonify({"predictions": predictions})

# Route for performance testing
@application.route('/performance_test', methods=['GET'])
def performance_test():
    test_case = "Justin Trudeau is the PM of Canada"  # Example test case

    # Open CSV file to write latency results
    with open('performance_test.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Test Case", "Start Time", "End Time", "Latency (ms)"])

        # Perform 100 API calls
        for i in range(100):
            start_time = time.time()
            
            # Send POST request
            response = requests.post(predict_link, json={'text': test_case})
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Write row to CSV
            writer.writerow([i+1, start_time, end_time, latency])

            print(f"Test {i+1}: Latency: {latency:.2f} ms")

    return "Performance test completed and results stored in performance_test.csv."


# Route to generate boxplot and calculate average latency
@application.route('/generate_boxplot', methods=['GET'])
def generate_boxplot():
    # Read the CSV file
    df = pd.read_csv('performance_test.csv')

    # Generate the boxplot
    plt.boxplot(df['Latency (ms)'])
    plt.title('Latency of 100 API Calls')
    plt.ylabel('Latency (ms)')
    plt.savefig('latency_boxplot.png')  # Save the plot as a PNG file
    plt.show()

    # Calculate the average latency
    average_latency = df['Latency (ms)'].mean()
    
    return jsonify({"average_latency": average_latency})


if __name__ == '__main__':
    application.run(port=8080, debug=True)
