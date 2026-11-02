"""In-process inference lifecycle and transport isolation."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from models.llama_cpp_backend import LlamaCppBackend
from models.vlm_backend import VLMBackendError, create_remote_vlm_tagger


def test_vision_logs_use_filtered_callback(monkeypatch):
    from models.llama_cpp_backend import _chat_handler

    class Handler:
        def __init__(self, **kwargs):
            self.options = kwargs

    callback = Mock()
    set_verbose = Mock()
    mtmd = SimpleNamespace(mtmd_log_set=Mock(), mtmd_helper_log_set=Mock())
    monkeypatch.setitem(sys.modules, 'llama_cpp', SimpleNamespace(mtmd_cpp=mtmd))
    monkeypatch.setitem(sys.modules, 'llama_cpp.llama_chat_format',
                        SimpleNamespace(MTMDChatHandler=Handler))
    monkeypatch.setitem(sys.modules, 'llama_cpp._logger', SimpleNamespace(
        llama_log_callback=callback, set_verbose=set_verbose,
    ))

    handler = _chat_handler('/model/mmproj.gguf', 256)

    set_verbose.assert_called_once_with(False)
    mtmd.mtmd_log_set.assert_called_once_with(callback, None)
    mtmd.mtmd_helper_log_set.assert_called_once_with(callback, None)
    assert handler.options['verbose'] is False


@pytest.fixture
def native(tmp_path, monkeypatch):
    model = Mock()
    model.create_chat_completion.return_value = {'choices': [{'message': {'content': '  建议  '}}]}
    constructor = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, 'llama_cpp', SimpleNamespace(Llama=constructor))
    handler = Mock()
    handler_factory = Mock(return_value=handler)
    monkeypatch.setattr('models.llama_cpp_backend._chat_handler', handler_factory)
    path = tmp_path / 'model.gguf'
    path.touch()
    config = {'model_path': str(path), 'mmproj_path': str(path), 'image_tokens': 256}
    return config, model, constructor, handler


def test_model_reused_image_passed_and_released(native, monkeypatch):
    config, model, constructor, handler = native
    monkeypatch.setattr('urllib.request.urlopen', Mock(side_effect=AssertionError('network used')))
    tagger = create_remote_vlm_tagger({'vlm_backend': {'type': 'llama_cpp', 'llama_cpp': config}})
    tagger.load()
    for _ in range(2):
        assert tagger.generate(Image.new('RGB', (8, 8)), 'Respond in Simplified Chinese.', 160) == '建议'
    constructor.assert_called_once()
    handler._init_mtmd_context.assert_called_once_with(model)
    args = model.create_chat_completion.call_args.kwargs
    assert args['messages'][0]['content'][0]['text'] == 'Respond in Simplified Chinese.'
    assert args['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,')
    assert args['max_tokens'] == 160
    assert constructor.call_args.kwargs['logits_all'] is False
    assert constructor.call_args.kwargs['n_threads'] == 4
    assert constructor.call_args.kwargs['main_gpu'] == -1
    assert constructor.call_args.kwargs['split_mode'] == 0
    assert constructor.call_args.kwargs['offload_kqv'] is False
    assert constructor.call_args.kwargs['op_offload'] is False
    tagger.unload()
    tagger.unload()
    model.close.assert_called_once()


def test_load_failure_closes_allocated_model_without_fallback(native):
    config, model, _, handler = native
    handler._init_mtmd_context.side_effect = ValueError('bad projector')
    with pytest.raises(VLMBackendError, match='bad projector'):
        LlamaCppBackend(config).load()
    model.close.assert_called_once()


def test_missing_file_is_explicit(native):
    config, _, constructor, _ = native
    config['mmproj_path'] = '/missing/projector.gguf'
    with pytest.raises(VLMBackendError, match='mmproj_path file not found'):
        LlamaCppBackend(config).load()
    constructor.assert_not_called()


def test_missing_dependency_is_explicit(monkeypatch):
    monkeypatch.setitem(sys.modules, 'llama_cpp', None)
    with pytest.raises(VLMBackendError, match='llama-cpp-python==0.3.36'):
        LlamaCppBackend({}).load()


def test_viewer_shutdown_releases_cached_model(monkeypatch):
    from api import model_cache

    tagger = Mock()
    monkeypatch.setattr(model_cache, '_vlm_tagger', tagger)
    model_cache.unload_vlm_tagger()
    assert model_cache._vlm_tagger is None
    tagger.unload.assert_called_once()
