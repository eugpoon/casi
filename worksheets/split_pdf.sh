# in terminal
# brew install pdftk-java ghostscript
# chmod +x split_pdf.sh 
# ./split_pdf.sh 

#!/bin/bash


# Path to your input PDF
INPUT_PDF="worksheets.pdf"

# Output directory (use the same directory as the input file if left empty)
OUTPUT_DIR=""

# Array of NASA center names for filenames
CENTERS=("AFRC" "AMES" "GISS" "GRC" "GSFC" "JPL" "JSC" "KSC" "LARC" "MAF" "MSFC" "SSC" "WFF" "WSTF")

# Get the total number of pages
TOTAL_PAGES=$(pdftk "$INPUT_PDF" dump_data | grep "NumberOfPages" | awk '{print $2}')

# Counter for centers
CENTER_INDEX=0

# Process every 2 pages
for ((i=1; i<=$TOTAL_PAGES; i+=2)); do
    # Calculate end page (making sure not to exceed total pages)
    END=$((i+1))
    if [ $END -gt $TOTAL_PAGES ]; then
        END=$TOTAL_PAGES
    fi
    
    # Get the current center name
    if [ $CENTER_INDEX -lt ${#CENTERS[@]} ]; then
        CENTER=${CENTERS[$CENTER_INDEX]}
    else
        # If we run out of center names, use numbers
        CENTER="Additional_$CENTER_INDEX"
    fi
    
    # Create output filename
    OUTPUT="${OUTPUT_DIR}${CENTER}.pdf"
    
    # Extract pages
    pdftk "$INPUT_PDF" cat $i-$END output "$OUTPUT"
    
    echo "Created $OUTPUT (pages $i-$END)"
    
    # Increment the center index
    CENTER_INDEX=$((CENTER_INDEX+1))
done