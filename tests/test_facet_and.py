"""Regression: AND facet triggers come from RAW goal text only."""
from cancer_output_atlas.rank import parse_goal, _facet_display_label


def _labels(goal: str) -> list[str]:
    return [_facet_display_label(f) for f in parse_goal(goal).required_facets]


def test_breast_rnaseq_facets():
    labs = _labels("breast cancer RNA-seq")
    assert "breast cancer" in labs and "rna-seq" in labs
    assert "NSCLC" not in labs


def test_lung_scrnaseq_not_nsclc_chip():
    labs = _labels("lung cancer single-cell RNA-seq")
    assert "lung cancer" in labs and "scRNA-seq" in labs
    assert "NSCLC" not in labs


def test_nsclc_alone_suppresses_broad_lung():
    labs = _labels("NSCLC")
    assert labs == ["NSCLC"] or ("NSCLC" in labs and "lung cancer" not in labs)


def test_nsclc_pembro_facets():
    labs = _labels(
        "Find public NSCLC pembrolizumab / Keytruda immunotherapy resources I can reuse"
    )
    assert "NSCLC" in labs
    assert "lung cancer" not in labs
    assert "pembrolizumab / Keytruda" in labs
    assert "immunotherapy" in labs


def test_nsclc_does_not_substring_sclc():
    labs = _labels("NSCLC RNA-seq")
    assert "NSCLC" in labs and "rna-seq" in labs
    assert "lung cancer" not in labs


def test_sclc_explicit_is_lung_cancer():
    labs = _labels("sclc rna-seq")
    assert "lung cancer" in labs and "rna-seq" in labs
    assert "NSCLC" not in labs


def test_melanoma_scrna():
    labs = _labels("melanoma single-cell RNA-seq")
    assert "melanoma" in labs and "scRNA-seq" in labs


def test_ovarian_cancer():
    labs = _labels("ovarian cancer RNA-seq")
    assert "ovarian cancer" in labs and "rna-seq" in labs


def test_keytruda_nsclc():
    labs = _labels("Keytruda NSCLC")
    assert "NSCLC" in labs and "pembrolizumab / Keytruda" in labs
    assert "lung cancer" not in labs


def test_scrna_not_or_bulk_rna():
    labs = _labels("lung cancer single-cell RNA-seq")
    assert "scRNA-seq" in labs and "rna-seq" not in labs


def test_broad_rnaseq_chip_not_scrna():
    labs = _labels("breast cancer RNA-seq")
    assert "breast cancer" in labs and "rna-seq" in labs
    assert "scRNA-seq" not in labs


def test_scrna_chip_label_strict():
    labs = _labels("lung cancer single-cell RNA-seq")
    assert labs.count("scRNA-seq") == 1
