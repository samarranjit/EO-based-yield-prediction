from farm_us.config import (
    BAND_ORDER,
    PAPER_BAND_MEAN,
    PRITHVI_BAND_MEAN,
    FarmConfig,
)


def test_band_order_is_prithvi_hls_order():
    assert BAND_ORDER == ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR1", "SWIR2")
    assert len(PAPER_BAND_MEAN) == 6
    assert len(PRITHVI_BAND_MEAN) == 6


def test_band_stats_mode_switch():
    cfg = FarmConfig()
    cfg.norm.mode = "fold_training_statistics"
    mean_paper, _ = cfg.band_stats()
    assert mean_paper == PAPER_BAND_MEAN
    cfg.norm.mode = "official_prithvi_statistics"
    mean_off, _ = cfg.band_stats()
    assert mean_off == PRITHVI_BAND_MEAN
