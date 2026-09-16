import pandas as pd

from reid.slices import slice_table


def test_slice_table_splits_queries_by_attribute():
    ap = pd.DataFrame({"image_id": ["q1", "q2"], "vehicle_id": [1, 2], "camera_id": [5, 89], "AP": [0.2, 0.8]})
    gallery = pd.DataFrame({"image_id": ["g1", "g2"], "vehicle_id": [1, 2], "camera_id": [6, 90]})
    attributes = pd.DataFrame({"image_id": ["q1", "q2", "g1", "g2"], "area_k": [100.0, 400.0, 0.0, 0.0],
                               "truncated": [False, False, False, False],
                               "taxi": ["private", "taxi", "private", "taxi"],
                               "view": ["front", "back", "back", "back"]})
    table = slice_table(ap, gallery, attributes).set_index(["slice", "group"]).mAP

    assert table[("cams_89_90", True)] == 0.8 and table[("cams_89_90", False)] == 0.2
    assert table[("view_in_other_cam", True)] == 0.8  # q2 is seen from the back by camera 90; q1's front is not there
    assert table[("view_in_other_cam", False)] == 0.2
    assert table[("taxi_livery", True)] == 0.8
