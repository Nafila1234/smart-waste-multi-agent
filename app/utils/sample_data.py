
import pandas as pd


def get_sample_bins():

    data = [
        {
            "bin_id": "BIN_001",
            "location": "City Center",
            "fill_level": 92,
            "waste_type": "Plastic",
            "priority": "HIGH",
            "status": "Collection Required"
        },
        {
            "bin_id": "BIN_002",
            "location": "University Area",
            "fill_level": 98,
            "waste_type": "Battery",
            "priority": "CRITICAL",
            "status": "Immediate Collection"
        },
        {
            "bin_id": "BIN_003",
            "location": "Residential Area",
            "fill_level": 35,
            "waste_type": "Paper",
            "priority": "LOW",
            "status": "Monitoring"
        },
        {
            "bin_id": "BIN_004",
            "location": "Commercial District",
            "fill_level": 86,
            "waste_type": "Organic",
            "priority": "HIGH",
            "status": "Collection Required"
        },
        {
            "bin_id": "BIN_005",
            "location": "Park Area",
            "fill_level": 65,
            "waste_type": "Glass",
            "priority": "MEDIUM",
            "status": "Scheduled"
        },
    ]

    return pd.DataFrame(data)
