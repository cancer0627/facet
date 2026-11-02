"""Process-local GGUFvision inference using llama-cpp-python."""

import logging
from time import perf_counter

from pathlib import Path
from threading import RLock

from models.vlm_backend import VLMBackend, VLMBackendError, _encode_jpeg_base64


def _chat_handler(mmproj_path, image_tokens):
    # Keep native imports lazy so other backends need no additional dependency.
    from llama_cpp.llama_chat_format import MTMDChatHandler
    from llama_cpp import mtmd_cpp
    from llama_cpp._logger import llama_log_callback, set_verbose

    # The vision library has separate native log sinks. Route both through
    # the binding's error-only logger instead of its default stderr output.
    # The imported callback stays alive in llama_cpp._logger for native use.
    set_verbose(False)
    mtmd_cpp.mtmd_log_set(llama_log_callback, None)
    mtmd_cpp.mtmd_helper_log_set(llama_log_callback, None)

    class FacetChatHandler(MTMDChatHandler):
        """Pin the projector budget and disable thinking in the model template."""

        def _init_mtmd_context(self, llama_model):
            if self.mtmd_ctx is not None:
                return
            mtmd = self._mtmd_cpp
            params = mtmd.mtmd_context_params_default()
            params.use_gpu = False
            params.print_timings = False
            params.n_threads = llama_model.n_threads
            params.image_min_tokens = image_tokens
            params.image_max_tokens = image_tokens
            self.mtmd_ctx = mtmd.mtmd_init_from_file(
                self.clip_model_path.encode(), llama_model.model, params,
            )
            if not self.mtmd_ctx:
                self.mtmd_ctx = None
                raise VLMBackendError('Could not load the vision projector')

            def close_projector():
                if self.mtmd_ctx is not None:
                    mtmd.mtmd_free(self.mtmd_ctx)
                    self.mtmd_ctx = None

            llama_model._stack.callback(close_projector)
            if not mtmd.mtmd_support_vision(self.mtmd_ctx):
                raise VLMBackendError('The configured model does not support images')

        def __call__(self, **kwargs):
            return super().__call__(**{**kwargs, 'enable_thinking': False})

    return FacetChatHandler(clip_model_path=mmproj_path, verbose=False, use_gpu=False)


logger = logging.getLogger(__name__)


class LlamaCppBackend(VLMBackend):
    """One lazily loaded model, shared serially for the lifetime of a job."""

    def __init__(self, config):
        self.config = dict(config)
        self._model = None
        self._lock = RLock()

    def load(self):
        with self._lock:
            if self._model is not None:
                return
            try:
                from llama_cpp import Llama
            except (ImportError, OSError) as ex:
                raise VLMBackendError(
                    'llama_cpp requires llama-cpp-python==0.3.36; '
                    'install the llama-cpp optional dependency. '
                    f'Native import failed: {ex}'
                ) from ex
            for key in ('model_path', 'mmproj_path'):
                path = self.config.get(key)
                if not path or not Path(path).expanduser().is_file():
                    raise VLMBackendError(f'vlm_backend.llama_cpp.{key} file not found: {path}')
            model = None
            try:
                handler = _chat_handler(
                    str(Path(self.config['mmproj_path']).expanduser()),
                    int(self.config.get('image_tokens', 256)),
                )
                model = Llama(
                    model_path=str(Path(self.config['model_path']).expanduser()),
                    chat_handler=handler,
                    n_ctx=int(self.config.get('n_ctx', 4096)),
                    n_threads=int(self.config.get('n_threads', 4)),
                    n_threads_batch=int(self.config.get('n_threads', 4)),
                    n_gpu_layers=0,
                    split_mode=0,  # LLAMA_SPLIT_MODE_NONE (no splitting across devices)
                    main_gpu=-1,  # Also exclude GPU devices when constructing the context.
                    offload_kqv=False,
                    op_offload=False,
                    # Critiques only need text generation; do not compute and retain logits for all input positions.
                    logits_all=False,
                    verbose=False,
                )
                handler._init_mtmd_context(model)
                self._model = model
            except Exception as ex:
                if model is not None:
                    model.close()
                raise VLMBackendError(f'Could not load llama_cpp model: {ex}') from ex

    def generate(self, image, prompt, max_new_tokens):
        with self._lock:
            started = perf_counter()
            cold = self._model is None
            self.load()
            loaded = perf_counter()
            logger.info("Photo model preparation took %.2fs, cold load=%s", loaded - started, cold)
            try:
                result = self._model.create_chat_completion(
                    messages=[{'role': 'user', 'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {
                            'url': f'data:image/jpeg;base64,{_encode_jpeg_base64(image)}',
                        }},
                    ]}],
                    temperature=0,
                    max_tokens=int(max_new_tokens),
                )
                logger.info(
                    "Photo model inference took %.2fs, usage %s",
                    perf_counter() - loaded, result.get('usage', {}),
                )
                choices = result.get('choices') or []
                if not choices:
                    return ''
                return ((choices[0].get('message') or {}).get('content') or '').strip()
            except Exception as ex:
                raise VLMBackendError(f'llama_cpp generation failed: {ex}') from ex

    def unload(self):
        with self._lock:
            if self._model is not None:
                try:
                    self._model.close()
                finally:
                    self._model = None
