# Define input and output directories
input_dir=$1   
output_dir=$2
var=$3

# Ensure output directory exists
mkdir -p "$output_dir"

# Loop through all matching files
for input_file in "$input_dir"/${var}_*.nc; do
    # Extract filename from the input path
    filename=$(basename "$input_file")
    
    # Define the output path
    output_file="$output_dir/$filename"
    
    # Perform the regridding
    cdo remapbil,grid_3x2.txt "$input_file" "$output_file"
    
    # Print status
    echo "Regridded: $input_file -> $output_file"
done