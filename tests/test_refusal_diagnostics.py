import numpy as np

from reid.metrics import refusal_curve


def test_official_diagnostics_keep_junk_top_answers_and_legacy_scores_distinct():
    q_pid, q_cam = np.arange(1, 7), np.full(6, 10)
    g_pid = np.array([1, 1, 2, 3, 4, 7, 8])
    g_cam = np.array([10, 11, 11, 11, 11, 11, 12])
    similarity = np.array([
        [0.9, 0.1, 0.8, 0.3, 0.2, 0.0, -0.1],  # junk top is officially right; the next item is wrong
        [0.1, 0.2, 0.9, 0.3, 0.4, 0.0, -0.1],
        [0.1, 0.2, 0.3, 0.8, 0.4, 0.9, -0.1],  # answered wrong match with a pair
        [0.1, 0.2, 0.3, 0.4, 0.9, 0.0, -0.1],  # refused although a pair exists
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.9, -0.1],  # answered despite no pair
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.9],   # refused with no pair
    ])
    row = refusal_curve(similarity, q_pid, q_cam, g_pid, g_cam,
                        confidence=np.array([5.0, 9.0, 9.0, 1.0, 9.0, 1.0]), grid=5.0).iloc[0]

    # A confidence equal to the threshold is answered. Wrong answered matches are FP, not FN.
    assert (row.TP_official, row.FP_official, row.FN_official, row.TN_official) == (2, 2, 1, 1)
    assert np.isclose(row.precision_official, 1 / 2)
    assert np.isclose(row.recall_official, 2 / 3)
    assert np.isclose(row.answered_fraction_official, 4 / 6)
    assert np.isclose(row.F1_official, 4 / 7) and row.TNR == 1 / 2
    assert np.isclose(row.score_official, 0.7 * (4 / 7) + 0.3 * (1 / 2))
    # The existing legacy and strict fields retain their original meaning and values.
    assert np.isclose(row.F1, 1 / 3) and np.isclose(row.F1_strict, 1 / 3)
    assert np.isclose(row.precision, 1 / 4) and np.isclose(row.recall, 1 / 2)
    assert np.isclose(row.score, 0.7 * (1 / 3) + 0.3 * (1 / 2))
    assert row.answered == row.answered_fraction_official


def test_official_diagnostics_handle_missing_denominators_and_row_subsets():
    from reid.metrics import official_refusal_diagnostics

    conf = np.array([5.0, 1.0])
    has_pair = np.array([False, False])
    right = np.array([False, False])
    all_rows = official_refusal_diagnostics(conf, has_pair, right, 5.0)
    assert all_rows == {"TP_official": 0, "FP_official": 1, "FN_official": 0, "TN_official": 1,
                        "precision_official": 0.0, "recall_official": 0.0,
                        "answered_fraction_official": 0.5}
    subset = official_refusal_diagnostics(conf, has_pair, right, 5.0, rows=np.array([False, True]))
    assert subset["TN_official"] == 1 and subset["FP_official"] == 0
    assert subset["precision_official"] == subset["recall_official"] == subset["answered_fraction_official"] == 0.0
