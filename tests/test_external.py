import zipfile

import pandas as pd

from reid.external import prepare


def test_prepare_keeps_present_images_prefixes_vehicles_and_unpacks(tmp_path):
    archive = tmp_path / "veriwild.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name in ("images/00005/000021.jpg", "images/00005/000022.jpg", "images/00007/000030.jpg"):
            z.writestr(name, b"jpeg")
        z.writestr("vehicle_info.txt",  # the first vehicle is not in the archive (train split)
                   "00001/000001;14;2018-03-07 16:59:23;Changan;SUV;white\n"
                   "00005/000021;3;2018-03-08 10:00:01;Audi;sedan;black\n"
                   "00005/000022;9;2018-03-08 10:05:12;Audi;sedan;black\n"
                   "00007/000030;3;2018-03-09 08:00:00;BYD;SUV;gray\n")
    table = prepare(archive, tmp_path / "out")

    assert table.image_id.tolist() == ["00005/000021", "00005/000022", "00007/000030"]
    assert table.vehicle_id.tolist() == ["vw_00005", "vw_00005", "vw_00007"]
    assert table.camera_id.tolist() == [3, 9, 3]
    assert (tmp_path / "out" / "images" / "00007" / "000030.jpg").read_bytes() == b"jpeg"
    # pandas 3 reads strings back with its own string dtype, so compare values only
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "out" / "train.csv"), table, check_dtype=False)
