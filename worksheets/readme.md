## Risk Worksheets

### TODO

* Add better data quality checks

---

### Contents

* `center_projections.ipynb`: Run this notebook to generate an Excel summary of risk data using input from a ZIP file.

---

### Extract Individual Center Worksheets from a Combined PDF

1. **Export Slides to PDF**
   Export all center worksheets from your slide deck (e.g., Google Slides or PowerPoint) into a single PDF file.

2. **Split the Combined PDF into Individual PDFs**

   * Assumptions:

     * Each NASA center has exactly **2 pages**.
     * The centers appear in the same order as listed in the `split_pdf.sh` script.

3. **Install Required Tools**
   Ensure `pdftk-java` and `ghostscript` are installed:

   ```bash
   brew install pdftk-java ghostscript
   ```

4. **Run the Split Script**

   ```bash
   chmod +x split_pdf.sh     # only needed once
   ./split_pdf.sh worksheets.pdf # xample file name
   ```

   This will create a folder named `worksheets_split/` and save one PDF per center inside it.