#!/bin/bash

# Ensure pdftk is installed
command -v pdftk >/dev/null 2>&1 || { echo >&2 "pdftk is required but not installed. Aborting. brew install pdftk-java ghostscript
"; exit 1; }

# Ask for input PDF if not provided
if [ -z "$1" ]; then
    read -p "Enter path to input PDF: " INPUT_PDF
else
    INPUT_PDF="$1"
fi

# Check if input file exists
if [ ! -f "$INPUT_PDF" ]; then
    echo "File '$INPUT_PDF' not found!"
    exit 1
fi

# Create output directory next to input PDF
BASENAME=$(basename "$INPUT_PDF" .pdf)
OUTPUT_DIR="$(dirname "$INPUT_PDF")/${BASENAME}_split"
mkdir -p "$OUTPUT_DIR"

# NASA centers
CENTERS=("AFRC" "AMES" "GISS" "GRC" "GSFC" "JPL" "JSC" "KSC" "LARC" "MAF" "MSFC" "SSC" "WFF" "WSTF")

# Get total number of pages
TOTAL_PAGES=$(pdftk "$INPUT_PDF" dump_data | grep "NumberOfPages" | awk '{print $2}')
CENTER_INDEX=0

# Split every 2 pages
for ((i=1; i<=$TOTAL_PAGES; i+=2)); do
    END=$((i+1 > TOTAL_PAGES ? TOTAL_PAGES : i+1))
    CENTER=${CENTERS[$CENTER_INDEX]:-Additional_$CENTER_INDEX}
    OUTPUT="${OUTPUT_DIR}/${CENTER}.pdf"
    pdftk "$INPUT_PDF" cat $i-$END output "$OUTPUT"
    echo "Created $OUTPUT (pages $i-$END)"
    CENTER_INDEX=$((CENTER_INDEX+1))
done
