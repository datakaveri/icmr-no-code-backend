# %%
import pandas as pd
import requests

class FHIRData:
    def __init__(self, base_url, group_id):
        self.group_id = group_id
        self.group_url = f"{base_url}/Group/{self.group_id}"
        self.patient_url = f"{base_url}/Patient/"
        self.patient_ids = self.fetch_patient_ids()

    def fetch_patient_ids(self):
        response = requests.get(self.group_url)
        response.raise_for_status()
        group_data = response.json()
        patient_ids = []
        for member in group_data.get('member', []):
            if 'entity' in member and member['entity']['reference'].startswith('Patient/'):
                patient_ids.append(member['entity']['reference'].split('/')[-1])
        return patient_ids
        
    def get_patient_data(self):
        patients_data = []
        for patient_id in self.patient_ids:
            patient_url = f"{self.patient_url}{patient_id}"
            response = requests.get(patient_url)
            response.raise_for_status()
            p = response.json()
            patients_data.append({
                'patient_id': p['id'],
                'gender': p.get('gender', 'unknown'),
                'active': p.get('active', False),
                'last_updated': p['meta']['lastUpdated']
            })
        return patients_data


# %%
class Patient:
    def __init__(self, patient_id, gender, active, last_updated):
        self.patient_id = patient_id
        self.gender = gender
        self.active = active
        self.last_updated = last_updated

# Define the PatientRepository class
class PatientRepository:
    def __init__(self, FHIRData):
        self.FHIRData = FHIRData
        self.patients = self._load_patients()

    def _load_patients(self):
        patients_data = self.FHIRData.get_patient_data()
        patients = []
        for pat_data in patients_data:
            patient = Patient(**pat_data)
            patients.append(patient)
        return patients

    def get_patients_dataframe(self):
        patients_data = []
        for patient in self.patients:
            patients_data.append({
                'patient_id': patient.patient_id,
                'gender': patient.gender,
                'active': patient.active,
                'last_updated': patient.last_updated
            })
        return pd.DataFrame(patients_data)

# Retrive Patient Data
# fhir = FHIRData()
# repository = PatientRepository(fhir)
# patients_df = repository.get_patients_dataframe()
# print(patients_df)

