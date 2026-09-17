import zipfile

import pandas as pd
import pytest

from reid.external import prepare, source_names


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


def test_veri776_takes_train_and_test_ids_and_cameras_from_file_names(tmp_path):
    archive = tmp_path / "veri776.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name in ("VeRi/image_train/0001_c001_00016450_0.jpg", "VeRi/image_train/0001_c014_00016510_1.jpg",
                     "VeRi/image_test/0002_c002_00030600_0.jpg",
                     "VeRi/image_query/0002_c002_00030600_0.jpg",  # repeats a test vehicle: skipped
                     "VeRi/YongtaiPoint_Google.jpg"):
            z.writestr(name, name.encode())
        z.writestr("VeRi/name_train.txt", "")
    table = prepare(archive, tmp_path / "out", "veri776")

    assert table.image_id.tolist() == ["image_test/0002_c002_00030600_0", "image_train/0001_c001_00016450_0",
                                       "image_train/0001_c014_00016510_1"]
    assert table.vehicle_id.tolist() == ["v776_0002", "v776_0001", "v776_0001"]
    assert table.camera_id.tolist() == [2, 1, 14]
    image = tmp_path / "out" / "images" / "image_train" / "0001_c014_00016510_1.jpg"
    assert image.read_bytes() == b"VeRi/image_train/0001_c014_00016510_1.jpg"
    assert not (tmp_path / "out" / "images" / "image_query").exists()


def test_vehicleid_keeps_labelled_present_images_with_one_camera_per_image(tmp_path):
    (tmp_path / "images").mkdir()
    for image in ("0000002", "0000003", "0000006"):  # 0000006 has no vehicle label
        (tmp_path / "images" / f"{image}.jpg").write_bytes(b"jpeg")
    (tmp_path / "img2vid.txt").write_text("0000003 00007\n0000002 00007\n0000009 00008\n")  # 0000009 is not unpacked
    table = prepare(None, tmp_path, "vehicleid")

    assert table.image_id.tolist() == ["0000002", "0000003"]
    assert table.vehicle_id.tolist() == ["vid_00007", "vid_00007"]
    assert table.camera_id.nunique() == 2
    assert (tmp_path / "train.csv").exists()


def test_vehicleid_must_be_unpacked_first(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare(None, tmp_path, "vehicleid")


def test_source_is_one_set_or_a_list():
    assert source_names({"source": "veriwild_test"}) == ["veriwild_test"]
    assert source_names({"source": ["veriwild_test", "veri776"]}) == ["veriwild_test", "veri776"]
