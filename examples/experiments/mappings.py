import importlib
import warnings

# name -> "module:Class". Imported on demand so that a sim-only machine (e.g. a headless GPU server
# without RealSense / SpaceMouse drivers) does not need the real-robot dependencies.
_CONFIGS = {
    "ram_insertion": "experiments.ram_insertion.config:TrainConfig",
    "usb_pickup_insertion": "experiments.usb_pickup_insertion.config:TrainConfig",
    "object_handover": "experiments.object_handover.config:TrainConfig",
    "egg_flip": "experiments.egg_flip.config:TrainConfig",
    "ur7e_sim_pick_place": "experiments.ur7e_sim_pick_place.config:TrainConfig",
}


class _LazyConfigMapping(dict):
    def __getitem__(self, name):
        module, cls = _CONFIGS[name].split(":")
        try:
            return getattr(importlib.import_module(module), cls)
        except ImportError as e:
            warnings.warn(f"Could not import config '{name}': {e}")
            raise

    def __contains__(self, name):
        return name in _CONFIGS

    def keys(self):
        return _CONFIGS.keys()


CONFIG_MAPPING = _LazyConfigMapping()
