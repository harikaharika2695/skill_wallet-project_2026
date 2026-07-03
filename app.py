from flask import Flask, render_template, request

app = Flask(__name__)

# Route for the Home Screen / Form Page
@app.route('/')
def home():
    # Renders the index.html form shown in your image
    return render_template('index.html')

# Route to Process the Form Submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1. Capture all form values using the HTML input names
        gender = request.form.get('gender')
        own_car = request.form.get('own_car')
        own_realestate = request.form.get('own_realestate')
        total_income = request.form.get('total_income')
        income_type = request.form.get('income_type')
        education = request.form.get('education')
        family_status = request.form.get('family_status')
        housing_type = request.form.get('housing_type')

        # 2. [Optional Placeholder] If you have a trained model (.pkl file), 
        # you would load it and pass these values into it here.
        
        # 3. Create a dummy result for now to test that the web pages connect
        # You can change this to "Approved" or "Rejected" based on your logic
        prediction_result = "Approved" 

        # Send the final result over to the result.html page
        return render_template('result.html', prediction=prediction_result)

if __name__ == '__main__':
    # Starts the local development server at http://127.0.0.1:5000/
    app.run(debug=True)
