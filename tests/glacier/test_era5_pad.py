"""The ERA5-Land request box must always span at least two grid points per axis."""

from pism_terra.glacier.climate import ERA5_LAND_PAD, pad_area

ERA5_LAND_DX = 0.1


def test_tiny_domain_gets_at_least_two_rows_and_columns():
    """A sub-cell domain still spans at least two ERA5-Land points per axis."""
    # Lat/lon box of RGI2000-v7.0-C-01-14094 (14 x 15 km at 58.4N): a bare
    # request returned one latitude row and two longitude columns.
    area = [58.33, -134.50, 58.47, -134.25]
    south, west, north, east = pad_area(area, pad=ERA5_LAND_PAD)
    assert (north - south) / ERA5_LAND_DX >= 2
    assert (east - west) / ERA5_LAND_DX >= 2


def test_pad_preserves_slot_order():
    """Padding keeps the caller's latitude slot order and moves West down, East up."""
    assert pad_area([60.0, -150.0, 59.0, -149.0], pad=0.5) == [60.5, -150.5, 58.5, -148.5]
    assert pad_area([59.0, -150.0, 60.0, -149.0], pad=0.5) == [58.5, -150.5, 60.5, -148.5]
