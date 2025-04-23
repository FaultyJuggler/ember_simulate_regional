import sys
import os

# Add your project and EMBER to the path
sys.path.append('/Users/felip/Developer/github/fed-reg-mal')
sys.path.append('/Users/felip/Developer/github/ember')

# Import the function and run it
from simulate_regional_split import split_ember_into_regions

# Use the function with the path to EMBER data
metadata = split_ember_into_regions(
    ember_data_path='/Users/felip/Developer/github/ember/data/ember2018/',
    output_dir='/Users/felip/Developer/github/fed-reg-mal/ember_data/',
    year='2018'
)

print(f"Created regional datasets with {metadata['total_samples']} total samples")