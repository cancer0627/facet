"""Isolate saliency scoring dependencies for older PyTorch versions."""
import atexit
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
from packaging.version import Version


BIREFNET_REVISION = '280306042f57b7a33854319da62fd86aaa89ec4c'
logger = logging.getLogger('facet.saliency_compat')


def local_model_directory():
    return Path(__file__).resolve().parents[1] / 'pretrained_models' / 'BiRefNet_dynamic' / BIREFNET_REVISION


def needs_compat_worker():
    if os.environ.get('FACET_SALIENCY_WORKER') == '1':
        return False
    return (Version(importlib.metadata.version('torch')) < Version('2.5')
            and Version(importlib.metadata.version('transformers')) >= Version('5'))


def compat_directory():
    return Path(sys.prefix) / 'saliency-compat'


def compat_available():
    return (compat_directory() / 'transformers' / '__init__.py').is_file()


class SaliencyWorker:
    """Reuse a model in a separate process, exchanging temporary images and numeric arrays."""

    def __init__(self, config):
        if not compat_available():
            raise ImportError('Saliency scoring requires compatible dependencies; run: '
                              f'{sys.executable} scripts/install_saliency_compat.py')
        root = str(Path(__file__).resolve().parents[1])
        env = dict(os.environ, FACET_SALIENCY_WORKER='1')
        env['PYTHONPATH'] = os.pathsep.join([str(compat_directory()), root])
        env['HF_MODULES_CACHE'] = str(compat_directory() / 'hf-modules')
        self._lock = threading.Lock()
        self.process = subprocess.Popen(
            [sys.executable, '-m', 'models.saliency_compat', json.dumps(config)],
            cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        try:
            self._read()
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Saliency compatibility worker exited; check the model loading logs.')
        result = json.loads(line)
        if not result.get('ok'):
            raise RuntimeError(result.get('error', 'Saliency scoring failed'))

    def predict(self, images, soft=False, batch_size=None):
        with self._lock, tempfile.TemporaryDirectory(prefix='facet-saliency-') as directory:
            paths = []
            started = time.perf_counter()
            logger.info('[主体分割耗时] 开始临时图片编码，共 %d 张', len(images))
            for index, img in enumerate(images):
                path = str(Path(directory) / f'{index}.png')
                img.save(path)
                paths.append(path)
            logger.info('[主体分割耗时] 临时图片编码与写入：%.2f 秒，%d 张，%.1f MiB（兆二进制字节）',
                        time.perf_counter() - started, len(paths),
                        sum(os.path.getsize(path) for path in paths) / (1024 * 1024))
            output = str(Path(directory) / 'masks.npz')
            request = dict(paths=paths, output=output, soft=soft, batch_size=batch_size)
            started = time.perf_counter()
            logger.info('[主体分割耗时] 开始等待子进程（含图片读取、模型计算和分割结果写入）')
            self.process.stdin.write(json.dumps(request) + '\n')
            self.process.stdin.flush()
            self._read()
            logger.info('[主体分割耗时] 子进程往返等待：%.2f 秒', time.perf_counter() - started)
            started = time.perf_counter()
            with np.load(output, allow_pickle=False) as arrays:
                masks = [arrays[f'arr_{index}'].copy() for index in range(len(images))]
            logger.info('[主体分割耗时] 分割结果读取：%.2f 秒', time.perf_counter() - started)
            return masks

    def close(self):
        with self._lock:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process.stdout.close()


def main():
    from contextlib import redirect_stdout
    import traceback
    from PIL import Image
    from models.saliency_scorer import SaliencyScorer

    protocol = sys.stdout

    def reply(**result):
        protocol.write(json.dumps(result) + '\n')
        protocol.flush()

    try:
        # Send all model and dependency output to the logs to keep process communication clean.
        with redirect_stdout(sys.stderr):
            scorer = SaliencyScorer(**json.loads(sys.argv[1]))
            scorer.load()
        reply(ok=True)
        for line in sys.stdin:
            request = json.loads(line)
            try:
                with redirect_stdout(sys.stderr):
                    images = []
                    for path in request['paths']:
                        with Image.open(path) as img:
                            images.append(img.convert('RGB'))
                    masks = ([scorer.get_saliency_soft(img) for img in images]
                             if request['soft'] else scorer.get_saliency_masks(
                                 images, batch_size=request['batch_size']))
                    np.savez(request['output'], *masks)
                reply(ok=True)
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                reply(ok=False, error=str(error))
        scorer.unload()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        reply(ok=False, error=str(error))


if __name__ == '__main__':
    main()
