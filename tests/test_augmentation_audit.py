from reid.augmentation_audit import audit_recipes


def test_audit_separates_each_transform_from_the_full_recipe():
    train = {"flip_p": .5, "pad": 16, "erasing_p": .25, "erasing_scale": [.02, .08],
             "erasing_ratio": [.5, 2.0],
             "color_jitter": {"p": .3, "brightness": .1, "contrast": .1, "saturation": .05, "hue": 0}}
    recipes = audit_recipes(train)
    assert recipes["original"]["flip_p"] == recipes["original"]["erasing_p"] == 0
    assert recipes["flip"]["flip_p"] == 1 and recipes["flip"]["pad"] == 0
    assert recipes["pad_crop"]["pad"] == 16 and recipes["pad_crop"]["erasing_p"] == 0
    assert recipes["color"]["color_jitter"]["p"] == 1
    assert recipes["erase"]["erasing_scale"] == [.02, .08]
    assert recipes["full"] is train
