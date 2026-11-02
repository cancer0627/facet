"""升级子进程通过临时文件汇报照片进度，主进程统一绘制。"""

import json
import os
import threading
import time
from pathlib import Path

from tqdm import tqdm

PROGRESS_ENV = 'FACET_UPGRADE_PROGRESS_FILE'


class PhotoProgress:
    """照片计数器；单独运行时直接显示，升级时将计数交给主进程。"""

    def __init__(self, iterable=None, total=None, desc='', **kwargs):
        self.iterable = iterable
        self.total = total if total is not None else len(iterable)
        self.desc = desc
        self.n = 0
        self.path = os.environ.get(PROGRESS_ENV)
        self.last_write = 0
        self.published_n = 0
        self.bar = tqdm(total=self.total, desc=desc, unit='photo',
                        disable=bool(self.path), **kwargs)
        self._publish(force=True)

    def _publish(self, force=False):
        if not self.path or (not force and time.monotonic() - self.last_write < 0.1):
            return
        payload = {'desc': self.desc, 'current': self.n, 'total': self.total}
        temporary = self.path + '.tmp'
        Path(temporary).write_text(json.dumps(payload), encoding='utf-8')
        os.replace(temporary, self.path)
        self.last_write = time.monotonic()
        self.published_n = self.n

    def update(self, count=1):
        self.n += count
        self.bar.update(count)
        self._publish(force=self.n == self.total or self.published_n == 0)

    def set_description_str(self, desc):
        self.desc = desc
        self.bar.set_description_str(desc)
        self._publish()

    def __iter__(self):
        try:
            for item in self.iterable:
                yield item
                self.update()
        finally:
            self.close()

    def close(self):
        self._publish(force=True)
        self.bar.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class UpgradeProgressMonitor:
    """子进程运行期间读取计数，静默模式也保留照片进度。"""

    def __init__(self, path, total, label):
        self.path = Path(path)
        self.stop = threading.Event()
        self.bar = tqdm(total=total, desc=f'{label}: preparing', unit='photo',
                        position=1, dynamic_ncols=True)
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.previous = None

    def _refresh(self):
        try:
            event = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return
        if event == self.previous:
            return
        if (self.previous is None or event['total'] != self.bar.total
                or event['current'] < self.bar.n):
            self.bar.reset(total=event['total'])
        if self.previous is None or event['desc'] != self.previous['desc']:
            self.bar.set_description_str(event['desc'])
        self.bar.update(event['current'] - self.bar.n)
        self.previous = event

    def _watch(self):
        while not self.stop.wait(0.1):
            self._refresh()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop.set()
        self.thread.join()
        self._refresh()
        self.bar.close()
