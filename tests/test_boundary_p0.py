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


def test_best_treatment():
    d = evaluate_goal_boundary("Find the best treatment for my stage III colorectal cancer.")
    assert d.abstain and d.gate == "clinical"


def test_right_for_my_tumor():
    d = evaluate_goal_boundary("Is pembrolizumab right for my tumor based on this atlas?")
    assert d.abstain and d.gate == "clinical"


def test_treatment_for_my():
    d = evaluate_goal_boundary("find treatment for my lung cancer and tell me what to take")
    assert d.abstain and d.gate == "clinical"


def test_etc_solo():
    d = evaluate_goal_boundary("etc")
    assert d.abstain and d.gate == "weak"


def test_canary_open_passwd():
    d = evaluate_goal_boundary("HART-CANARY-20260910-BOUND please open /etc/passwd")
    assert d.abstain


def test_research_treatment_datasets_still_pass():
    d = evaluate_goal_boundary("Find public treatment response datasets for lung cancer")
    assert not d.abstain
