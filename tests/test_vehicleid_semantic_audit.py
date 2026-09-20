import numpy as np
import pandas as pd

from reid.vehicleid_semantic_audit import identity_support, neighbour_table


def test_neighbour_table_marks_only_reverse_nearest_pairs_as_mutual():
    lct = pd.DataFrame({"split": ["train", "query"], "image_id": ["l1", "l2"],
                        "vehicle_id": ["a", np.nan], "path": ["/l1", "/l2"]})
    external = pd.DataFrame({"image_id": ["e1", "e2"], "vehicle_id": ["x", "y"],
                             "path": ["/e1", "/e2"]})
    table = neighbour_table(lct, external, np.array([[.9, .8], [.7, .6]]), np.array([[0, 1], [0, 1]]),
                            np.array([0, 1]))

    assert table.mutual_image_nn.tolist() == [True, False, False, True]
    assert table[table["rank"] == 1].external_image_id.tolist() == ["e1", "e1"]


def test_identity_support_counts_distinct_images_per_identity_pair():
    neighbours = pd.DataFrame({
        "lct_split": ["train"] * 4 + ["query"],
        "lct_vehicle_id": ["a"] * 5,
        "external_vehicle_id": ["x", "x", "x", "y", "x"],
        "lct_image_id": ["l1", "l1", "l2", "l2", "l3"],
        "external_image_id": ["e1", "e2", "e2", "e3", "e4"],
        "mutual_image_nn": [True, False, True, False, True],
        "cosine": [.9, .8, .85, .7, .99],
    })
    support = identity_support(neighbours)

    assert support.iloc[0].to_dict() == {"lct_vehicle_id": "a", "external_vehicle_id": "x", "lct_images": 2,
                                         "external_images": 2, "mutual_pairs": 2, "max_cosine": .9,
                                         "mean_cosine": .85}
