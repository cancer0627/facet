"""用可控时钟验证扫描耗时累计，以及中断时的统计保留。"""

import logging
from types import SimpleNamespace

import pytest

from processing import multi_pass
from processing import resource_monitor


@pytest.fixture
def timed_processor(monkeypatch):
    clock = [0.0]

    def advance(seconds, result=None):
        clock[0] += seconds
        return result

    monkeypatch.setattr(multi_pass, '_ensure_imports', lambda: None)
    monkeypatch.setattr(multi_pass.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(multi_pass.ModelManager, 'detect_accelerator', lambda: None)
    manager = SimpleNamespace(
        detect_vram=lambda: 0,
        get_active_profile=lambda: {},
        load_model_only=lambda name: advance(2, object()),
        unload_model=lambda *a, **kw: advance(1),
        unload_all=lambda: advance(1),
        device='cpu', _cache_hits=0, _cache_misses=0,
    )
    processor = multi_pass.ChunkedMultiPassProcessor(None, manager)
    processor.chunk_size = 1
    processor.pass_groups = [['clip']]
    monkeypatch.setattr(processor, '_load_images',
                        lambda paths: advance(1, {p: {} for p in paths}))
    monkeypatch.setattr(processor, '_run_model_pass', lambda *args: advance(3))
    monkeypatch.setattr(processor, '_compute_aggregates', lambda *args: advance(4))
    monkeypatch.setattr(processor, '_save_results', lambda *args: advance(5))
    monkeypatch.setattr(processor, '_reclaim_freed_memory', lambda: advance(1))
    monitor = SimpleNamespace(start=lambda: None, stop=lambda: None,
                              note_chunk_complete=lambda: None, adjustments=[])
    monkeypatch.setattr(resource_monitor, 'MultiPassResourceMonitor', lambda *args: monitor)
    return processor, advance


def test_scan_stage_totals_cover_completed_chunks(timed_processor, caplog):
    processor, _ = timed_processor
    with caplog.at_level(logging.INFO, logger='facet.multi_pass'):
        metrics = processor.process_directory(['a', 'b'], show_progress=False)

    timings = metrics['stage_timings']
    assert timings['模型 clip 加载（含缓存恢复）'] == {'seconds': 4, 'calls': 2}
    assert timings['模型 clip 计算（含预处理与数据传递）'] == {'seconds': 6, 'calls': 2}
    assert timings['综合评分计算']['seconds'] == 8
    assert timings['结果保存与提交']['seconds'] == 10
    assert metrics['images_processed'] == 2
    assert sum(t['seconds'] for t in timings.values()) == metrics['total_time'] == 37
    assert '第 2 批结束' in caplog.text
    assert '分阶段累计（按耗时降序）' in caplog.text


def test_interrupted_stage_keeps_time_and_prints_summary(timed_processor, monkeypatch, caplog):
    processor, advance = timed_processor

    def interrupt(*args):
        advance(7)
        raise KeyboardInterrupt

    monkeypatch.setattr(processor, '_run_model_pass', interrupt)
    with caplog.at_level(logging.INFO, logger='facet.multi_pass'), pytest.raises(KeyboardInterrupt):
        processor.process_directory(['a'], show_progress=False)

    timings = processor.metrics['stage_timings']
    assert timings['模型 clip 计算（含预处理与数据传递）']['seconds'] == 7
    assert '结果保存与提交' not in timings
    assert processor.metrics['total_time'] == 11
    assert '中断或异常：模型 clip 计算' in caplog.text
    assert '分阶段累计（按耗时降序）' in caplog.text
