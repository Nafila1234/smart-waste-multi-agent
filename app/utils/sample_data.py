
import pandas as pd


def get_sample_bins():

    data = [
        {
            "bin_id": "BIN_001",
            "location": "City Center",
            "latitude": 36.7665,
            "longitude": 3.4772,
            "fill_level": 92,
            "waste_type": "Plastic",
            "priority": "HIGH",
            "status": "Collection Required"
        },
        {
            "bin_id": "BIN_002",
            "location": "University Area",
            "latitude": 36.7680,
            "longitude": 3.4850,
            "fill_level": 98,
            "waste_type": "Battery",
            "priority": "CRITICAL",
            "status": "Immediate Collection"
        },
        {
            "bin_id": "BIN_003",
            "location": "Residential Area",
            "latitude": 36.7600,
            "longitude": 3.4700,
            "fill_level": 35,
            "waste_type": "Paper",
            "priority": "LOW",
            "status": "Monitoring"
        },
        {
            "bin_id": "BIN_004",
            "location": "Commercial District",
            "latitude": 36.7550,
            "longitude": 3.4800,
            "fill_level": 86,
            "waste_type": "Organic",
            "priority": "HIGH",
            "status": "Collection Required"
        },
        {
            "bin_id": "BIN_005",
            "location": "Park Area",
            "latitude": 36.7730,
            "longitude": 3.4650,
            "fill_level": 65,
            "waste_type": "Glass",
            "priority": "MEDIUM",
            "status": "Scheduled"
        },
    ]

    return pd.DataFrame(data)
