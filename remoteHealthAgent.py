import csv
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

class RemoteHealthAgent:
    def __init__(self):
        """
        Initialize the RemoteHealthAgent class.
        """
        self.symptom_data = {}  # Dictionary to store patient symptom data
        self.experience_data = {}  # Dictionary to store historical symptom analysis data
        self.disease_graph = {}  # Graph representing relationships between symptoms and diseases
        self.encoder = OneHotEncoder()  # One-hot encoder for symptoms

    def read_csv_file(self, filename):
        """
        Read data from a CSV file and collect patient symptoms.
        """
        with open(filename, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                patient_id = row['patient_id']
                symptoms = row['symptoms'].split(',')
                location = row['location']  # Extract location from the CSV
                self.collect_symptoms(patient_id, symptoms, location)

    def collect_symptoms(self, patient_id, symptoms, location):
        """
        Collect symptoms from a patient and store them in the symptom data dictionary.
        """
        self.symptom_data[patient_id] = {'symptoms': symptoms, 'location': location}

    def gather_experience(self):
        """
        Gather experience from the analyzed symptom data to enhance the agent's knowledge.
        """
        for patient_id, data in self.symptom_data.items():
            # Extract symptoms and location for each patient
            symptoms = data['symptoms']
            location = data['location']

            # Assuming a simple learning mechanism for gathering experience
            # Update or create entries in the experience_data dictionary based on analyzed symptoms
            if patient_id not in self.experience_data:
                self.experience_data[patient_id] = {}
            for symptom in symptoms:
                if symptom not in self.experience_data[patient_id]:
                    self.experience_data[patient_id][symptom] = 1
                else:
                    self.experience_data[patient_id][symptom] += 1

    def build_disease_graph(self):
        """
        Build a graph representing relationships between symptoms and diseases.
        This graph can be used for basic disease detection.
        """
        # Placeholder for building the disease graph - Customize based on actual data
        self.disease_graph = {
            'fever': ['Flu', 'Malaria', 'COVID-19'],
            'cough': ['Flu', 'Pneumonia', 'COVID-19'],
            'shortness of breath': ['Pneumonia', 'Asthma', 'COVID-19'],
            'headache': ['Migraine', 'Sinusitis', 'COVID-19'],
            'fatigue': ['Flu', 'Anemia', 'COVID-19'],
            'nausea': ['Flu', 'Food poisoning', 'COVID-19'],
            'vomiting': ['Food poisoning', 'Gastroenteritis', 'COVID-19'],
            'diarrhea': ['Food poisoning', 'Gastroenteritis', 'COVID-19'],
            'muscle pain': ['Flu', 'Fibromyalgia', 'COVID-19'],
            'chills': ['Flu', 'Malaria', 'COVID-19'],  
        }

    def analyze_symptoms(self, patient_id):
        """
        Analyze symptoms for disease detection.
        """
        if not self.disease_graph:
            self.build_disease_graph()

        patient_data = self.symptom_data.get(patient_id)
        if not patient_data:
            return f"Location: Unknown\nPatient {patient_id}: No symptoms recorded."

        patient_symptoms = patient_data['symptoms']
        location = patient_data['location']

        detected_diseases, probability = self.detect_disease_with_ml(patient_symptoms)

        if detected_diseases:
            most_likely_disease = detected_diseases[0]  # Assuming the first detected disease is the most likely
            
            # Calculate disease severity based on the number of matching symptoms
            severity = self.calculate_severity(detected_diseases, patient_symptoms)

            return f"Location: {location}\nDetected diseases: {', '.join(detected_diseases)}.\nMost likely disease: {most_likely_disease}.\nProbability: {probability:.2f}\nSymptoms: {', '.join(patient_symptoms)}\nSeverity: {severity}"
        else:
            return f"Location: {location}\nNo specific disease detected."
        
    def calculate_severity(self, detected_diseases, patient_symptoms):
        """
        Calculate the severity of the disease based on the number of matching symptoms.
        """
        max_match_count = 0
        for disease in detected_diseases:
            match_count = sum(symptom in patient_symptoms for symptom in self.disease_graph.get(disease, []))
            max_match_count = max(max_match_count, match_count)

        if max_match_count >= 3:
            return "Severe"
        elif max_match_count >= 1:
            return "Affected"
        else:
            return "Normal"

    def detect_disease_with_ml(self, symptoms):
        """
        Use machine learning model to detect disease based on symptoms.
        """
        # Fit the encoder with all unique symptoms encountered during analysis
        all_symptoms = list(set(symptom for data in self.symptom_data.values() for symptom in data['symptoms']))
        self.encoder.fit([[symptom] for symptom in all_symptoms])

        # One-hot encode the symptoms
        encoded_symptoms = self.encoder.transform([[symptom] for symptom in symptoms]).toarray()

        # Generate input features based on all possible symptoms
        input_features = [1 if symptom in symptoms else 0 for symptom in self.disease_graph.keys()]

        # Initialize and train the decision tree classifier (for demonstration purposes)
        model = DecisionTreeClassifier()
        model.fit([input_features] * len(encoded_symptoms), encoded_symptoms)

        # Make predictions using the trained model
        predicted_disease_prob = model.predict_proba([input_features])[0]  # Assuming only one disease is predicted
        max_probability = np.max(predicted_disease_prob)

        # Detect diseases based on a probability threshold
        detected_diseases = []
        for prob, disease in zip(predicted_disease_prob, self.disease_graph.values()):
            if np.any(prob > 0.5):  # Considering a probability threshold for disease detection
                detected_diseases.extend(disease)

        detected_diseases = list(set(detected_diseases))  # Remove duplicates

        return detected_diseases, max_probability

if __name__ == "__main__":
    agent = RemoteHealthAgent()

    # Read data from CSV file
    agent.read_csv_file("patient_data.csv")

    # Gather experience from the collected symptom data
    agent.gather_experience()

    # Take user input for patient ID
    patient_id = input("Enter patient ID (P001-P299): ")

    # Analyze symptoms for the provided patient ID
    result_patient = agent.analyze_symptoms(patient_id)

    # Display result for the patient
    print(result_patient)
