# list_standard_types.py

from chembl_webresource_client.new_client import new_client
import logging


def list_available_standard_types(target_chembl_id, limit=1000):
    """
    List available standard types for a target.

    Parameters:
        target_chembl_id (str): The ChEMBL ID of the target.

    Returns:
        set: Set of unique standard_types.
    """
    activity = new_client.activity
    filters = {"target_chembl_id": target_chembl_id}
    activities = (
        activity.filter(**filters).only(["standard_type"]).only_unique("standard_type")
    )
    activities = activities[:limit]
    standard_types = set()
    for act in activities:
        standard_type = act.get("standard_type")
        if standard_type:
            standard_types.add(standard_type)
    return standard_types


if __name__ == "__main__":
    import sys

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Example usage:
    target_id = "CHEMBL25"  # Replace with your target's ChEMBL ID
    standard_types = list_available_standard_types(target_id, limit=1000)
    print(f"Available standard_types for {target_id}:")
    for st in sorted(standard_types):
        print(f"- {st}")
