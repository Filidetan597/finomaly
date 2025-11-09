import pandas as pd
from fpdf import FPDF

class PDFReporter:
    def __init__(self):
        pass

    def generate_pdf_report(self, df, output_path, figs=None):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
    # Table headers
        for col in df.columns:
            pdf.cell(40, 10, str(col), 1)
        pdf.ln()
        pdf.set_font('Arial', '', 10)
    # Table rows
        for _, row in df.iterrows():
            for col in df.columns:
                pdf.cell(40, 10, str(row[col]), 1)
            pdf.ln()
    # If figure(s) are provided, add them to the PDF
        if figs is not None:
            import tempfile
            import os
            if not isinstance(figs, list):
                figs = [figs]
            for fig in figs:
                tmpfile = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                fig.savefig(tmpfile.name, format='png')
                tmpfile.close()
                pdf.add_page()
                pdf.image(tmpfile.name, x=10, y=10, w=180)
                os.unlink(tmpfile.name)
        pdf.output(output_path)
        return output_path
