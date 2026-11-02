"""Install isolated compatible saliency dependencies without overwriting the main environment."""
from pathlib import Path
import argparse
import os
import subprocess
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-model', action='store_true', help='Also download the pinned saliency model')
    parser.add_argument('--endpoint', default='https://huggingface.co', help='Model download endpoint')
    args = parser.parse_args()
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '--no-deps', '--upgrade',
        '--target', str(Path(sys.prefix) / 'saliency-compat'),
        'transformers==4.57.6', 'huggingface-hub==0.36.0', 'tokenizers==0.22.2',
    ], check=True)
    if args.download_model:
        root = str(Path(__file__).resolve().parents[1])
        env = dict(os.environ, HF_ENDPOINT=args.endpoint, HF_HUB_DISABLE_XET='1')
        env['PYTHONPATH'] = os.pathsep.join([str(Path(sys.prefix) / 'saliency-compat'), root])
        subprocess.run([sys.executable, '-c',
                        'from huggingface_hub import snapshot_download; '
                        'from models.saliency_compat import BIREFNET_REVISION, local_model_directory; '
                        'snapshot_download("ZhengPeng7/BiRefNet_dynamic", revision=BIREFNET_REVISION, '
                        'local_dir=str(local_model_directory()), '
                        'allow_patterns=["config.json", "BiRefNet_config.py", "birefnet.py", "model.safetensors"])'],
                       env=env, cwd=root, check=True)
