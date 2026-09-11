from cancer_output_atlas.find import _card_summary
from cancer_output_atlas.schema import Identifier, OutputRecord


def _rec(summary: str) -> OutputRecord:
    return OutputRecord(
        output_id="dataset:geo:GSE295685",
        kind="dataset",
        title="TNG260, a novel small-molecule CoREST inhibitor, sensitizes STK11-mutant tumors to anti-PD-1 immunotherapy",
        summary=summary,
        identifiers=[Identifier(scheme="geo", value="GSE295685", source="fixture")],
        landing_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE295685",
        source_status="fixture",
    )


SUMMARY = (
    "Non-small cell lung cancer (NSCLC) patients with loss of the tumor suppressor "
    "gene STK11 are resistant to immune checkpoint therapies like anti-PD-1. Our "
    "in-vivo CRISPR screen identified HDAC1 as a target that, when inhibited, "
    "reversed anti-PD-1 resistance driven by loss of STK11. We developed TNG260, "
    "a potent small-molecule inhibitor of the CoREST complex. In the tumors of "
    "patients with STK11-deficient cancers, treatment with a combination of TNG260 "
    "and pembrolizumab (NCT05887492) increased intratumoral histone acetylation."
)


def test_card_summary_shows_pembro_when_chip_requires_it():
    rec = _rec(SUMMARY)
    lead = _card_summary(rec)
    assert "pembrolizumab" not in lead.lower()
    shown = _card_summary(
        rec, prefer_terms=["NSCLC", "pembrolizumab / Keytruda", "immunotherapy"]
    )
    assert "pembrolizumab" in shown.lower()
