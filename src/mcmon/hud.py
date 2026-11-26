"""Hide the on-screen inventory so a monitor cannot read the label off the frame."""
import numpy as np

# Regions as fractions of the frame: (top, bottom, left, right).
HUD_REGIONS = (
    (0.82, 1.00, 0.00, 1.00),   # hotbar, health, chat text
    (0.00, 0.28, 0.72, 1.00),   # pick-up and recipe pop-ups (room for three)
)
MODES = ("masked", "hud_only", "none")


def apply_hud_mode(frames: np.ndarray, mode: str = "masked") -> np.ndarray:
    """[T, H, W, 3] frames with the HUD blacked out, or everything but the HUD."""
    if mode == "none":
        return frames
    if mode not in MODES:
        raise ValueError(f"unknown HUD mode {mode!r}")
    h, w = frames.shape[1:3]
    hud = np.zeros((h, w), dtype=bool)
    for top, bottom, left, right in HUD_REGIONS:
        hud[int(top * h):int(np.ceil(bottom * h)), int(left * w):int(np.ceil(right * w))] = True
    keep = hud if mode == "hud_only" else ~hud
    return frames * keep[None, :, :, None].astype(frames.dtype)
