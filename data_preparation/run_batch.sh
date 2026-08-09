#!/bin/bash

STATES=('DE' 'IA' 'IL' 'IN' 'KS' 'MI' 'MN' 'MO' 'ND' 'NE' 'OH' 'PA' 'SD' 'WI')
YEARS=(2020 2021 2022 2023 2024)
PYTHON_SCRIPT="/home/cholab/LabMembers/Samar/EO-based-yield-prediction/data_preparation/scripts/merge_pseudo_countydistributed_yield_data_for_states.py"

for state in "${STATES[@]}"; do
    for year in "${YEARS[@]}"; do
        echo "=================================================="
        echo "Processing State: $state | Year: $year"
        echo "=================================================="
        
        python "$PYTHON_SCRIPT" --state "$state" --year "$year"
        
        if [ $? -ne 0 ]; then
            echo "FAILED: $state - $year"
        fi
    done
done

echo "Finished processing all state and year combinations."
