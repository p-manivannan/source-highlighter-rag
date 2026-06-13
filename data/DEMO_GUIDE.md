# Equifax Enterprise RAG Case File

This dataset contains five complete public records about the 2017 Equifax data
breach. Together they support enterprise research across cybersecurity,
governance, regulatory enforcement, contracting, outsourcing, and customer
remediation.

## Documents

1. `01_house_oversight_equifax_breach_report.pdf` - congressional root-cause report
2. `02_gao_equifax_response_report.pdf` - Equifax and federal customer responses
3. `03_ftc_equifax_complaint.pdf` - alleged unreasonable security practices
4. `04_ftc_equifax_stipulated_order.pdf` - binding remediation and governance duties
5. `05_fca_equifax_final_notice_2023.pdf` - UK outsourcing and customer-treatment findings

The files are unmodified public documents. Their publishers, source URLs, page
counts, and SHA-256 checksums are recorded in `dataset_manifest.json`.

## Recommended Demo Flow

```powershell
uv run python ingest.py
uv run streamlit run app.py
```

Then use `demo_questions.json`. A strong sequence is:

1. Ask whether the breach depended on a zero-day vulnerability.
2. Ask which controls allowed expansion and delayed detection.
3. Compare FTC allegations with the safeguards imposed by the order.
4. Ask how federal customers changed contracts and identity proofing.
5. Ask what UK failures led to the FCA penalty.

This demonstrates fact checking, cross-document synthesis, legal obligation
extraction, and clickable page-level evidence.

## Refreshing the Dataset

```powershell
uv run python scripts/download_equifax_dataset.py
```

GAO's CDN may reject automated clients. The manifest therefore records a
byte-preserving Internet Archive retrieval URL for the exact official GAO asset,
while retaining the GAO product page as the authoritative source.
