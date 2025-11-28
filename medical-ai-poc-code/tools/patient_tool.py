# tools/patient_tool.py

import os
import json
from typing import List, Dict
from langchain.tools import tool

# patients.json is expected at: project_root/data/patients.json
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "patients.json"
)


def _load_patients() -> List[Dict]:
    """Internal helper: load patients from JSON. Returns [] if file missing/invalid."""
    if not os.path.exists(DATA_PATH):
        return []
    try:
        with open(DATA_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


@tool("get_patient_report")
def get_patient_report(patient_name: str) -> str:
    """
    Look up a patient's discharge summary by their name.

    Input: patient_name (string)
    Output: short text with summary or a not-found message.
    """
    patients = _load_patients()
    matches = [
        p for p in patients
        if p.get("name", "").lower() == patient_name.strip().lower()
    ]

    if not matches:
        return f"No discharge summary found for patient '{patient_name}'."

    patient = matches[0]
    summary = patient.get("discharge_summary") or patient.get("summary") or "No summary field available."
    mrn = patient.get("mrn", "N/A")
    return (
        f"Patient: {patient.get('name', 'Unknown')}\n"
        f"MRN: {mrn}\n"
        f"Discharge Summary:\n{summary}"
    )


@tool("list_patients")
def list_patients(_: str = "") -> str:
    """
    List all known patients in the system.

    Input: ignored.
    Output: bullet list of patients.
    """
    patients = _load_patients()
    if not patients:
        return "No patients found in the system."

    lines = []
    for p in patients:
        name = p.get("name", "Unknown")
        mrn = p.get("mrn", "N/A")
        lines.append(f"- {name} (MRN: {mrn})")

    return "Known patients:\n" + "\n".join(lines)
