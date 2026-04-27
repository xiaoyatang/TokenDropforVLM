from typing import Any, Dict, List, Optional, Union
from .vision_compressor import VisionCompressor

def build_vision_compressor(compressive_cfg: Optional[Any] = None):
    """
    Builder for a standalone VisionCompressor module.
    model_cfg: used to read embed_dim (hidden size) and num_smb (summary bank size, must >0 when using compress).
    compressive_cfg: compressive configuration dict (same format as ViTCompress.compressive_cfg)
    """
    cfg = _maybe_to_dict(compressive_cfg) if compressive_cfg is not None else None
    return VisionCompressor(cfg)


# ---------------------------------------------------------------------- #
# Builder + param-group helper
# ---------------------------------------------------------------------- #
def _maybe_to_dict(cfg: Any) -> Dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return cfg
