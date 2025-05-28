from typing_extensions import TypedDict

class MedicalRecordFormatterState(TypedDict):
    input: str
    medical_record_template: str
    formatted_medical_record: str