import csv
import random

# Sample symptoms
symptoms_list = ["fever", "cough", "shortness of breath", "headache", "fatigue", "nausea", "muscle pain"]

# Generate sample data for 150 patients
patients = []
for i in range(1, 300):
    patient_id = f"P{i:03}"
    symptoms = random.sample(symptoms_list, random.randint(1, len(symptoms_list)))  # Randomly select symptoms for each patient
    symptoms_str = ",".join(symptoms)
    location = f"Remote Location {random.randint(1, 10)}"
    patients.append({"Patient_ID": patient_id, "Symptoms": symptoms_str, "Location": location})

# Define the fieldnames for the CSV file
fieldnames = ["patient_id", "symptoms", "location"]

# Generate the CSV file
csv_file = "patient_data.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write the patient data
    for patient in patients:
        writer.writerow(patient)

print(f"CSV file '{csv_file}' generated successfully.")
