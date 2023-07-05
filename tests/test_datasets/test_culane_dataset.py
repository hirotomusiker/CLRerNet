from libs.datasets import CulaneDataset


def test_culane_dataset():
    data_root = "tests/data"
    data_list = "tests/data/culane_dummy_list.txt"
    pipeline = []
    dataset = CulaneDataset(data_root, data_list, pipeline, test_mode=True)
    assert dataset.img_infos == ["fake.jpg"]
