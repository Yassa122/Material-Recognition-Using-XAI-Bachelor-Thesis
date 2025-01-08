# fetch_chembl_data.py

from chembl_webresource_client.new_client import new_client
import pandas as pd
import math


def fetch_binding_affinity(target_chembl_id, assay_type="IC50", limit=10000):
    """
    Fetch binding affinity data from ChEMBL for a specified target.

    Parameters:
        target_chembl_id (str): The ChEMBL ID of the target (e.g., 'CHEMBL25' for EGFR).
        assay_type (str): The type of assay to filter (default is 'IC50').
        limit (int): Maximum number of records to fetch.

    Returns:
        pd.DataFrame: DataFrame containing SMILES and pIC50.
    """
    activity = new_client.activity
    filters = {"target_chembl_id": target_chembl_id, "standard_type": assay_type}

    logging.info(
        f"Fetching activities for target {target_chembl_id} with assay type {assay_type}..."
    )
    activities = activity.filter(**filters).only(
        ["canonical_smiles", "standard_value", "standard_units"]
    )
    activities = activities[:limit]

    data = []
    count = 0  # Counter for fetched records
    for act in activities:
        smiles = act.get("canonical_smiles")
        value = act.get("standard_value")
        units = act.get("standard_units")

        if smiles and value and units:
            try:
                value = float(value)
            except ValueError:
                continue  # Skip entries with non-numeric values

            # Convert to pIC50 if units are nM or µM
            if units.lower() in ["n", "nm", "nanomolar"]:
                pic50 = -math.log10(value * 1e-9)  # pIC50
            elif units.lower() in ["u", "um", "micromolar"]:
                pic50 = -math.log10(value * 1e-6)  # pIC50
            else:
                pic50 = None  # Unsupported units

            if pic50:
                data.append({"SMILES": smiles, "pIC50": pic50})
                count += 1

    logging.info(f"Fetched {count} binding affinity records.")
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    import logging

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Example usage:
    target_id = "CHEMBL25"  # Replace with your target's ChEMBL ID
    df_binding = fetch_binding_affinity(target_id, assay_type="IC50", limit=5000)
    df_binding.to_csv("binding_affinity_data.csv", index=False)
    print(
        f"Fetched {len(df_binding)} binding affinity records and saved to 'binding_affinity_data.csv'."
    )
