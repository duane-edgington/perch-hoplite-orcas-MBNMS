"""src/torch_model.py
TensorFlow mock injection and Perch V2 PyTorch model loading for the
Perch-Hoplite pipeline.

The perch-hoplite library imports TensorFlow unconditionally at module load
time. This module injects a minimal TF mock so the library imports succeed
without TF installed, then loads the native PyTorch Perch V2 model.
"""
import importlib.machinery
import logging
import os
import sys
import types

log = logging.getLogger(__name__)


def inject_tf_mock() -> bool:
    """Inject a minimal TensorFlow mock into sys.modules if TF is not installed.

    This allows perch_hoplite.agile.classifier to import successfully without
    TensorFlow. The mock satisfies all module-level imports but raises clear
    errors if any TF functionality is actually called at runtime.

    Returns True if the mock was injected, False if real TF is already present.
    """
    if 'tensorflow' in sys.modules:
        return False

    tf_mock = types.ModuleType('tensorflow')
    tf_mock.__spec__ = importlib.machinery.ModuleSpec('tensorflow', loader=None)
    tf_mock.__version__ = '0.0.0-mock'
    tf_mock.Tensor = object

    keras_mock = types.ModuleType('tensorflow.keras')
    keras_mock.__spec__ = importlib.machinery.ModuleSpec('tensorflow.keras', loader=None)
    keras_mock.Model = object
    keras_mock.layers = types.ModuleType('tensorflow.keras.layers')
    keras_mock.optimizers = types.ModuleType('tensorflow.keras.optimizers')
    keras_mock.losses = types.ModuleType('tensorflow.keras.losses')
    tf_mock.keras = keras_mock

    sys.modules['tensorflow'] = tf_mock
    sys.modules['tensorflow.keras'] = keras_mock
    sys.modules['tensorflow.keras.layers'] = keras_mock.layers
    sys.modules['tensorflow.keras.optimizers'] = keras_mock.optimizers
    sys.modules['tensorflow.keras.losses'] = keras_mock.losses
    return True


def load_model_from_db(db, cuda_available_fn=None):
    """Load the Perch V2 embedding model for a given DB.

    For DBs created by the native PyTorch pipeline (model_key='perch_torch')
    or the Colab TF pipeline (model_key='taxonomy_model_tf'), loads the
    PerchTorchModel from ~/perch-pytorch — no TensorFlow needed.

    Falls back to the TF model_configs path only if PERCH_USE_TF=1 is set.

    Parameters
    ----------
    db : HopliteDBInterface
        An open Hoplite database.
    cuda_available_fn : callable | None
        Function returning True if CUDA is available. If None, uses a
        simple torch.cuda.is_available() check.

    Returns
    -------
    (embedding_model, audio_sources)
    """
    from perch_hoplite.agile import source_info

    db_model_config = db.get_metadata("model_config")
    embed_config    = db.get_metadata("audio_sources")
    audio_sources   = source_info.AudioSources.from_config_dict(embed_config)
    model_key       = db_model_config.model_key

    use_tf = os.environ.get("PERCH_USE_TF", "0") == "1"

    if cuda_available_fn is None:
        def cuda_available_fn():
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False

    if not use_tf and model_key in ("taxonomy_model_tf", "perch_torch"):
        pytorch_dir = os.path.expanduser("~/perch-pytorch")
        if pytorch_dir not in sys.path:
            sys.path.insert(0, pytorch_dir)
        try:
            from perch_hoplite_torch_adapter import PerchTorchModel
            embedding_model = PerchTorchModel(
                weights_dir=os.path.join(pytorch_dir, "perch_weights"),
                exact_mel=os.path.join(pytorch_dir, "const__pad1_output_0.npy"),
                device="cuda" if cuda_available_fn() else "cpu",
            )
            log.info("Loaded embedding model: PerchTorchModel (PyTorch, no TF)")
        except ImportError as e:
            log.warning("PerchTorchModel not available (%s); falling back to TF", e)
            use_tf = True

    if use_tf:
        from perch_hoplite.zoo import model_configs
        model_class = model_configs.get_model_class(model_key)
        embedding_model = model_class.from_config(db_model_config.model_config)
        log.info("Loaded embedding model: %s (TensorFlow)", model_key)

    return embedding_model, audio_sources
