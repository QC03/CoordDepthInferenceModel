from pose_lifting.filtering import OneEuroFilter


def test_filter_reset_on_gap_and_track_isolation():
    f = OneEuroFilter(max_dt=.2)
    assert f.filter(1., 0., 1) == 1.
    assert f.filter(3., .1, 2) == 3.
    assert f.filter(9., 1., 1) == 9.
