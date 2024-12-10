from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import psycopg2
import uuid

app = Flask(__name__)

# Database querying function
def query_data():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="treatment",
            user="postgres",
            password="lamis"
        )
        query = """
            SELECT 
            uuid, facility, hospital_number, age, gender, education_level, marital_status, art_duration, 
            changed_regimen, side_effects, adherence, missed_doses, base_line_viral_load, 
            current_viral_load, most_recent_viral_load, first_cd4, current_cd4, smoking, alcohol, 
            recreational_drugs, experience, clinic_appointments, barriers
            FROM treatment_data_new
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()  # Return an empty dataframe in case of an error


# Preprocessing function for clustering
def preprocess_data_for_clustering(df):
    categorical_columns = ['facility', 'gender', 'education_level', 'marital_status', 'changed_regimen',
                           'side_effects', 'adherence', 'base_line_viral_load', 'current_viral_load',
                           'most_recent_viral_load', 'first_cd4', 'current_cd4', 'smoking', 'alcohol',
                           'recreational_drugs', 'experience', 'clinic_appointments', 'barriers', 'missed_doses']
    
    missed_doses_mapping = {'0': 0, '1-2': 1, '3-5': 2, '>5': 3}
    df['missed_doses'] = df['missed_doses'].map(missed_doses_mapping).fillna(-1)
    
    encoders = {}
    for column in categorical_columns:
        if column in df.columns:
            encoder = LabelEncoder()
            df[column] = df[column].astype(str)
            df[column] = encoder.fit_transform(df[column])
            encoders[column] = encoder
    
    features = df.drop(columns=['hospital_number', 'uuid'])  # Excluding uuid and hospital_number
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return df, scaled_features, encoders, scaler


# Function to explain clusters
def explain_clusters(cluster_id):
    explanations = {
        0: "Cluster 0: Patients with high adherence, low viral load, and stable CD4 count.",
        1: "Cluster 1: Patients who may have fluctuating viral load and experience side effects.",
        2: "Cluster 2: Patients with poor adherence, high viral load, and potentially more barriers to treatment.",
        3: "Cluster 3: Patients with a high chance of interrupting treatment due to poor adherence and frequent missed doses."
    }
    return explanations.get(cluster_id, "No explanation available")


# Home route
@app.route('/')
def index():
    return render_template('index.html')


# Questionnaire route
@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if request.method == 'POST':
        # Get data from form submission
        form_data = {
            "facility": request.form['facility'],
            "hospital_number": request.form['hospital_number'],
            "age": request.form['age'],
            "gender": request.form['gender'],
            "education_level": request.form['education_level'],
            "marital_status": request.form['marital_status'],
            "art_duration": request.form['art_duration'],
            "changed_regimen": request.form['changed_regimen'],
            "side_effects": request.form['side_effects'],
            "adherence": request.form['adherence'],
            "missed_doses": request.form['missed_doses'],
            "base_line_viral_load": request.form['base_line_viral_load'],
            "current_viral_load": request.form['current_viral_load'],
            "most_recent_viral_load": request.form['most_recent_viral_load'],
            "first_cd4": request.form['first_cd4'],
            "current_cd4": request.form['current_cd4'],
            "smoking": request.form['smoking'],
            "alcohol": request.form['alcohol'],
            "recreational_drugs": request.form['recreational_drugs'],
            "experience": request.form['experience'],
            "clinic_appointments": request.form['clinic_appointments'],
            "barriers": request.form['barriers']
        }
        save_data_to_db(form_data)  # Call function to save data into the database
        return render_template('questionnaire.html', success=True)

    return render_template('questionnaire.html')


# Function to save data into PostgreSQL with UUID
def save_data_to_db(data):
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="treatment",
            user="postgres",
            password="lamis"
        )
        cursor = conn.cursor()
        
        # Insert query with uuid
        query = """
            INSERT INTO treatment_data_new (
                uuid, facility, hospital_number, age, gender, education_level, marital_status, 
                art_duration, changed_regimen, side_effects, adherence, missed_doses, 
                base_line_viral_load, current_viral_load, most_recent_viral_load, first_cd4, 
                current_cd4, smoking, alcohol, recreational_drugs, experience, 
                clinic_appointments, barriers
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Generate a UUID for the entry
        unique_id = str(uuid.uuid4())
        
        # Execute the query
        cursor.execute(query, (
            unique_id, data['facility'], data['hospital_number'], data['age'], 
            data['gender'], data['education_level'], data['marital_status'], data['art_duration'], 
            data['changed_regimen'], data['side_effects'], data['adherence'], 
            data['missed_doses'], data['base_line_viral_load'], data['current_viral_load'], 
            data['most_recent_viral_load'], data['first_cd4'], data['current_cd4'], 
            data['smoking'], data['alcohol'], data['recreational_drugs'], data['experience'], 
            data['clinic_appointments'], data['barriers']
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        return str(e)


# Prediction route
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Fetch data from the database
        data = query_data()
        if data.empty:
            return "No data available for prediction."

        # Preprocess the data for clustering
        data, scaled_features, encoders, scaler = preprocess_data_for_clustering(data)

        # Run the KMeans clustering model
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(scaled_features)
        data['cluster'] = kmeans.predict(scaled_features)

        # Reverse label encoding for the 'facility' column to show the actual name
        for column, encoder in encoders.items():
            data[column] = encoder.inverse_transform(data[column])

        # Add the cluster explanation
        data['cluster_explanation'] = data['cluster'].apply(lambda x: explain_clusters(x))

        # Select only the relevant columns for display
        data_to_display = data[['uuid', 'hospital_number', 'facility', 'cluster', 'cluster_explanation']]

        # Remove unwanted newline characters
        data_to_display = data_to_display.applymap(lambda x: str(x).replace('\n', '').strip())

        # Ensure the data to be displayed is passed correctly to the template
        return render_template('prediction.html', tables=[data_to_display.to_html(classes='table table-striped', header="true")])

    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
