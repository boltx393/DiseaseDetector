import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from collections import Counter

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

    def append_to_csv(self, filename, patient_id, symptoms, location):
        """
        Append new patient data to the CSV file.
        """
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([patient_id, ','.join(symptoms), location])

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

        detected_diseases, disease_probabilities = self.calculate_disease_probability(patient_symptoms)

        if detected_diseases:
            # Format detected diseases and their probabilities
            detected_diseases_output = ", ".join([f"{disease} ({disease_probabilities[disease]:.2f})" for disease in detected_diseases])

            # Determine the most likely disease based on highest probability
            most_likely_disease = max(disease_probabilities, key=disease_probabilities.get)
            
            # Calculate disease severity based on the number of matching symptoms
            severity = self.calculate_severity(detected_diseases, patient_symptoms)

            return f"Location: {location}\nDetected diseases: {detected_diseases_output}.\nMost likely disease: {most_likely_disease}\nSymptoms: {', '.join(patient_symptoms)}\nSeverity: {severity}"
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

        # Detect diseases based on a probability threshold
        detected_diseases = []
        for prob, disease in zip(predicted_disease_prob, self.disease_graph.values()):
            if np.any(prob > 0.3):  # Considering a probability threshold for disease detection
                detected_diseases.extend(disease)

        detected_diseases = list(set(detected_diseases))  # Remove duplicates

        return detected_diseases

    def calculate_disease_probability(self, symptoms):
        """
        Calculate the probabilities of all detected diseases based on the observed symptoms.
        """
        if not self.disease_graph:
            self.build_disease_graph()

        # Count occurrences of each symptom in the historical data
        symptom_counter = Counter()
        for data in self.symptom_data.values():
            symptom_counter.update(data['symptoms'])

        # Calculate the total number of patients
        total_patients = len(self.symptom_data)

        # Calculate the probability of each disease based on observed symptoms
        disease_probability = {}
        detected_diseases = set()
        for symptom in symptoms:
            if symptom in self.disease_graph:
                for disease in self.disease_graph[symptom]:
                    detected_diseases.add(disease)
                    # Calculate the probability of each disease given the symptom
                    disease_probability[disease] = (disease_probability.get(disease, 0) + (symptom_counter[symptom] + 1) / (total_patients + len(symptom_counter)))

        # Normalize probabilities
        total_probability = sum(disease_probability.values())
        for disease in disease_probability:
            disease_probability[disease] /= total_probability

        # Return the detected diseases and their probabilities
        return list(detected_diseases), disease_probability

if __name__ == "__main__":
    agent = RemoteHealthAgent()

    # Read data from CSV file
    agent.read_csv_file("patient_data.csv")

    # Gather experience from the collected symptom data
    agent.gather_experience()

    print("Options:")
    print("1. Add new patient")
    print("2. Retrieve patient details")

    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        # Determine the next patient ID
        next_patient_id = max((int(patient_id[1:]) for patient_id in agent.symptom_data.keys() if patient_id.startswith("P")), default=299) + 1
        if next_patient_id < 300:
            next_patient_id = 300
        next_patient_id = f"P{next_patient_id:03d}"

        # Take user input for new patient data
        patient_id = next_patient_id
        symptoms = input("Enter patient symptoms (comma-separated): ").split(',')
        location = input("Enter remote location: ")

        # Append new patient data to CSV file
        agent.append_to_csv("patient_data.csv", patient_id, symptoms, location)

        # Gather experience from the updated data
        agent.gather_experience()

        print("Patient data added successfully.")

    elif choice == "2":
        # Display the range of patient IDs
        min_patient_id = min((int(patient_id[1:]) for patient_id in agent.symptom_data.keys() if patient_id.startswith("P")), default=300)
        max_patient_id = max((int(patient_id[1:]) for patient_id in agent.symptom_data.keys() if patient_id.startswith("P")), default=299)
        print(f"Patient ID Range: P{min_patient_id:03d} - P{max_patient_id:03d}")

        # Take user input for patient ID
        patient_id = input("Enter patient ID: ")

        # Analyze symptoms for the provided patient ID
        result_patient = agent.analyze_symptoms(patient_id)

        # Display result for the patient
        print(result_patient)

    else:
        print("Invalid choice.")
