from cancer_output_atlas.boundary import evaluate_goal_boundary, clinical_intent


def test_clinical_diagnose():
    d = evaluate_goal_boundary("diagnose my patient stage III lung cancer treatment plan")
    assert d.abstain and d.gate == "clinical"


def test_clinical_prescribe():
    d = evaluate_goal_boundary("what drug should I prescribe for melanoma")
    assert d.abstain and d.gate == "clinical"


def test_noncancer_ra():
    d = evaluate_goal_boundary("Find public rheumatoid arthritis RNA-seq datasets")
    assert d.abstain and d.gate == "noncancer"


def test_noncancer_diabetes():
    d = evaluate_goal_boundary("non-cancer diabetes scRNA resources")
    assert d.abstain and d.gate == "noncancer"


def test_weak_bound():
    d = evaluate_goal_boundary("bound")
    assert d.abstain and d.gate == "weak"


def test_weak_open():
    d = evaluate_goal_boundary("open")
    assert d.abstain and d.gate == "weak"


def test_tp_breast_rnaseq_passes_gate():
    d = evaluate_goal_boundary("breast cancer RNA-seq")
    assert not d.abstain


def test_nonsense_not_clinical():
    assert not clinical_intent("qzwxecrvtbynulmpkj424242")
