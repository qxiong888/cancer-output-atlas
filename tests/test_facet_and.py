from cancer_output_atlas.rank import parse_goal


def test_breast_rnaseq_facets():
    q = parse_goal("breast cancer RNA-seq")
    assert q.required_facets
    flat = " ".join(" ".join(f) for f in q.required_facets)
    assert "breast" in flat and "rna" in flat


def test_nsclc_pembro_facets():
    q = parse_goal(
        "Find public NSCLC pembrolizumab / Keytruda immunotherapy resources I can reuse"
    )
    assert len(q.required_facets) >= 3
    joined = [" ".join(f) for f in q.required_facets]
    assert any("nsclc" in j or "lung" in j for j in joined)
    assert any("pembrolizumab" in j or "keytruda" in j for j in joined)
    assert any("immunotherapy" in j for j in joined)
