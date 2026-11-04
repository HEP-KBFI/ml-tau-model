from enum import IntEnum


class TauIDClass(IntEnum):
    BACKGROUND = 0
    HADRONIC_TAU = 1
    TAU_ELECTRON = 2
    TAU_MUON = 3


SIGNAL_CLASS_NAMES = tuple(member.name.lower() for member in TauIDClass if member != TauIDClass.BACKGROUND)


def tau_id_classes(cfg):
    """Return ordered class names and sample-to-background-class indices."""
    groups = cfg.dataset.get("background_classes", None)
    if groups is None:
        groups = {"background": ["qq"]}
    if not groups or "background" not in groups or next(iter(groups)) != "background":
        raise ValueError("dataset.background_classes must start with 'background'.")

    names = [member.name.lower() for member in TauIDClass]
    samples = {}
    for group, members in groups.items():
        if group in SIGNAL_CLASS_NAMES or (group != "background" and group in names):
            raise ValueError(f"Invalid tau-ID background class: {group}")
        if group != "background":
            names.append(group)
        if not members:
            raise ValueError(f"Background class {group} has no samples.")
        for sample in members:
            if sample == "z" or sample in samples:
                raise ValueError(f"Invalid or repeated background sample: {sample}")
            samples[str(sample)] = names.index(group)
    return tuple(names), samples