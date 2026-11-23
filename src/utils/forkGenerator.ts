import { ACCENT_HEX, BRAND_NAME, REPO_README_URL, REPO_URL } from '../data/defaults.ts';
import type {
  DownloadModelItem,
  ForkConfig,
  VoiceModelProject,
} from '../types/studio.ts';
import {
  buildTrainPipelineCommands,
  inferMatchingDUrl,
  sanitizeModelName,
} from './rvcTuning.ts';
import { buildStreamlitHeaderAndTabsPart1 } from './streamlitWebuiPart1.ts';
import { buildStreamlitTabsPart2 } from './streamlitWebuiPart2.ts';
import { buildStreamlitTabsPart3 } from './streamlitWebuiPart3.ts';

export interface GeneratedBundleFile {
  id: string;
  filename: string;
  zipPath: string;
  language: 'json' | 'python' | 'bash';
  description: string;
  content: string;
}

export function generateThemePython(): string {
  return `[theme]
base = "dark"
primaryColor = "${ACCENT_HEX}"
backgroundColor = "#0D0405"
secondaryBackgroundColor = "#160709"
textColor = "#FDF7F7"
font = "sans serif"

[server]
headless = true
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 2048

[browser]
gatherUsageStats = false
`;
}

export function generateAppPython(
  config: ForkConfig,
  project: VoiceModelProject,
  downloads: DownloadModelItem[] = []
): string {
  return (
    buildStreamlitHeaderAndTabsPart1(config, project, downloads) +
    buildStreamlitTabsPart2() +
    buildStreamlitTabsPart3()
  );
}

export function generateTurboTrainerPython(): string {
  return `from __future__ import annotations

import atexit
from concurrent.futures import Future, ThreadPoolExecutor
import inspect
import io
import os
from pathlib import Path
import threading
from typing import Any

_APPLIED = False
_CACHE_LOCK = threading.Lock()
_ARRAY_CACHE: dict[str, Any] = {}
_CACHE_BYTES = 0
_CACHE_CHECKS = 0
_PENDING_SAVES: dict[str, Future[None]] = {}
_SAVE_LOCK = threading.Lock()
_SAVE_POOL: ThreadPoolExecutor | None = None

_DATASET_SUBDIRS = ("sliced_audios", "extracted", "f0", "f0_voiced")
_MAX_SINGLE_FILE_BYTES = 16 * 1024 * 1024
_PER_PROCESS_RAM_CAP_BYTES = 2 * 1024 * 1024 * 1024
_MIN_SYSTEM_FREE_RAM_BYTES = 8 * 1024 * 1024 * 1024


def _read_mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except Exception:
        pass
    return 0


def _has_safe_ram(incoming_bytes: int) -> bool:
    global _CACHE_CHECKS
    if _CACHE_BYTES + incoming_bytes > _PER_PROCESS_RAM_CAP_BYTES:
        return False
    _CACHE_CHECKS += 1
    if _CACHE_CHECKS == 1 or _CACHE_CHECKS % 24 == 0:
        avail = _read_mem_available_bytes()
        if avail > 0 and avail - incoming_bytes < _MIN_SYSTEM_FREE_RAM_BYTES:
            return False
    return True


def _is_cacheable_dataset_path(path: Any) -> str | None:
    if not isinstance(path, (str, os.PathLike)):
        return None
    raw = os.fspath(path)
    norm = raw.replace("\\\\", "/")
    if "/logs/" not in norm and not norm.startswith("logs/"):
        return None
    if not any(f"/{part}/" in norm or norm.startswith(f"{part}/") for part in _DATASET_SUBDIRS):
        return None
    try:
        if os.path.getsize(raw) > _MAX_SINGLE_FILE_BYTES:
            return None
    except OSError:
        return None
    return raw


def _apply_cuda_tuning() -> None:
    import torch

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True


def _patch_dataloader() -> None:
    import torch.utils.data

    orig_init = torch.utils.data.DataLoader.__init__
    if getattr(orig_init, "_noregrets_patched", False):
        return

    cpu_cap = max(2, min(4, os.cpu_count() or 2))

    def fast_init(self: Any, *args: Any, **kwargs: Any) -> Any:
        num_workers = kwargs.get("num_workers", 0)
        if "num_workers" not in kwargs and len(args) >= 6:
            num_workers = args[5]
        if isinstance(num_workers, int) and num_workers > 0:
            if "num_workers" in kwargs and num_workers > cpu_cap:
                kwargs["num_workers"] = cpu_cap
            kwargs.setdefault("pin_memory", True)
            kwargs.setdefault("persistent_workers", True)
            kwargs.setdefault("prefetch_factor", 4)
        try:
            return orig_init(self, *args, **kwargs)
        except (TypeError, ValueError):
            kwargs.pop("persistent_workers", None)
            kwargs.pop("prefetch_factor", None)
            kwargs.pop("pin_memory", None)
            return orig_init(self, *args, **kwargs)

    fast_init._noregrets_patched = True  # type: ignore[attr-defined]
    torch.utils.data.DataLoader.__init__ = fast_init  # type: ignore[method-assign]


def _patch_optimizer() -> None:
    import torch

    orig_adamw_init = torch.optim.AdamW.__init__
    if not getattr(orig_adamw_init, "_noregrets_patched", False):
        supports_fused = "fused" in inspect.signature(orig_adamw_init).parameters

        def fast_adamw_init(self: Any, params: Any, *args: Any, **kwargs: Any) -> Any:
            _apply_cuda_tuning()
            if supports_fused and torch.cuda.is_available() and "fused" not in kwargs:
                if not isinstance(params, (list, tuple)):
                    params = list(params)
                try:
                    return orig_adamw_init(self, params, *args, fused=True, **kwargs)
                except Exception:
                    pass
            return orig_adamw_init(self, params, *args, **kwargs)

        fast_adamw_init._noregrets_patched = True  # type: ignore[attr-defined]
        torch.optim.AdamW.__init__ = fast_adamw_init  # type: ignore[method-assign]

    orig_zero_grad = torch.optim.Optimizer.zero_grad
    if not getattr(orig_zero_grad, "_noregrets_patched", False):
        def fast_zero_grad(self: Any, set_to_none: bool = True) -> Any:
            return orig_zero_grad(self, set_to_none=set_to_none)

        fast_zero_grad._noregrets_patched = True  # type: ignore[attr-defined]
        torch.optim.Optimizer.zero_grad = fast_zero_grad  # type: ignore[method-assign]


def _patch_ddp() -> None:
    import torch.nn.parallel

    ddp_cls = torch.nn.parallel.DistributedDataParallel
    orig_ddp_init = ddp_cls.__init__
    if getattr(orig_ddp_init, "_noregrets_patched", False):
        return

    def fast_ddp_init(self: Any, *args: Any, **kwargs: Any) -> Any:
        _apply_cuda_tuning()
        kwargs.setdefault("gradient_as_bucket_view", True)
        try:
            return orig_ddp_init(self, *args, **kwargs)
        except TypeError:
            kwargs.pop("gradient_as_bucket_view", None)
            return orig_ddp_init(self, *args, **kwargs)

    fast_ddp_init._noregrets_patched = True  # type: ignore[attr-defined]
    ddp_cls.__init__ = fast_ddp_init  # type: ignore[method-assign]


def _patch_tensorboard_images() -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return

    def _skip_image(*args: Any, **kwargs: Any) -> None:
        return None

    for method in ("add_image", "add_images", "add_figure"):
        if hasattr(SummaryWriter, method):
            setattr(SummaryWriter, method, _skip_image)


def _patch_ram_cache() -> None:
    import numpy as np

    orig_np_load = np.load
    if not getattr(orig_np_load, "_noregrets_patched", False):
        def fast_np_load(file: Any, *args: Any, **kwargs: Any) -> Any:
            global _CACHE_BYTES
            if args or kwargs:
                return orig_np_load(file, *args, **kwargs)
            key = _is_cacheable_dataset_path(file)
            if key is None:
                return orig_np_load(file)
            cache_key = f"np::{key}"
            with _CACHE_LOCK:
                cached = _ARRAY_CACHE.get(cache_key)
            if cached is not None:
                return cached.copy()
            arr = orig_np_load(file)
            nbytes = int(getattr(arr, "nbytes", 0))
            if nbytes > 0 and isinstance(arr, np.ndarray):
                with _CACHE_LOCK:
                    if cache_key not in _ARRAY_CACHE and _has_safe_ram(nbytes):
                        _ARRAY_CACHE[cache_key] = arr.copy()
                        _CACHE_BYTES += nbytes
            return arr

        fast_np_load._noregrets_patched = True  # type: ignore[attr-defined]
        np.load = fast_np_load  # type: ignore[assignment]

    try:
        import scipy.io.wavfile as wavfile

        orig_wav_read = wavfile.read
        if not getattr(orig_wav_read, "_noregrets_patched", False):
            def fast_wav_read(filename: Any, *args: Any, **kwargs: Any) -> Any:
                global _CACHE_BYTES
                if args or kwargs:
                    return orig_wav_read(filename, *args, **kwargs)
                key = _is_cacheable_dataset_path(filename)
                if key is None:
                    return orig_wav_read(filename)
                cache_key = f"wav::{key}"
                with _CACHE_LOCK:
                    cached = _ARRAY_CACHE.get(cache_key)
                if cached is not None:
                    sr, data = cached
                    return sr, data.copy()
                sr, data = orig_wav_read(filename)
                nbytes = int(getattr(data, "nbytes", 0))
                if nbytes > 0:
                    with _CACHE_LOCK:
                        if cache_key not in _ARRAY_CACHE and _has_safe_ram(nbytes):
                            _ARRAY_CACHE[cache_key] = (sr, data.copy())
                            _CACHE_BYTES += nbytes
                return sr, data

            fast_wav_read._noregrets_patched = True  # type: ignore[attr-defined]
            wavfile.read = fast_wav_read  # type: ignore[assignment]
    except Exception:
        pass

    try:
        import soundfile as sf

        orig_sf_read = sf.read
        if not getattr(orig_sf_read, "_noregrets_patched", False):
            def fast_sf_read(file: Any, *args: Any, **kwargs: Any) -> Any:
                global _CACHE_BYTES
                if args or kwargs:
                    return orig_sf_read(file, *args, **kwargs)
                key = _is_cacheable_dataset_path(file)
                if key is None:
                    return orig_sf_read(file)
                cache_key = f"sf::{key}"
                with _CACHE_LOCK:
                    cached = _ARRAY_CACHE.get(cache_key)
                if cached is not None:
                    data, sr = cached
                    return data.copy(), sr
                data, sr = orig_sf_read(file)
                nbytes = int(getattr(data, "nbytes", 0))
                if nbytes > 0:
                    with _CACHE_LOCK:
                        if cache_key not in _ARRAY_CACHE and _has_safe_ram(nbytes):
                            _ARRAY_CACHE[cache_key] = (data.copy(), sr)
                            _CACHE_BYTES += nbytes
                return data, sr

            fast_sf_read._noregrets_patched = True  # type: ignore[attr-defined]
            sf.read = fast_sf_read  # type: ignore[assignment]
    except Exception:
        pass


def _write_atomic(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(payload)
        os.replace(tmp, target)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def _patch_async_save() -> None:
    global _SAVE_POOL
    import torch

    orig_save = torch.save
    orig_load = torch.load
    if getattr(orig_save, "_noregrets_patched", False):
        return

    _SAVE_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nr-ckpt")
    atexit.register(lambda: _SAVE_POOL.shutdown(wait=True) if _SAVE_POOL else None)

    def fast_save(obj: Any, f: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            _SAVE_POOL is not None
            and isinstance(f, (str, os.PathLike))
            and not args
            and not kwargs
        ):
            raw = os.fspath(f)
            norm = raw.replace("\\\\", "/")
            if norm.endswith(".pth") and ("/logs/" in norm or norm.startswith("logs/")):
                buf = io.BytesIO()
                orig_save(obj, buf)
                target = Path(raw).resolve()
                fut = _SAVE_POOL.submit(_write_atomic, target, buf.getvalue())
                with _SAVE_LOCK:
                    _PENDING_SAVES[str(target)] = fut
                return None
        return orig_save(obj, f, *args, **kwargs)

    def safe_load(f: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(f, (str, os.PathLike)):
            key = str(Path(os.fspath(f)).resolve())
            with _SAVE_LOCK:
                fut = _PENDING_SAVES.get(key)
            if fut is not None:
                fut.result()
        return orig_load(f, *args, **kwargs)

    fast_save._noregrets_patched = True  # type: ignore[attr-defined]
    torch.save = fast_save  # type: ignore[assignment]
    torch.load = safe_load  # type: ignore[assignment]


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True
    for step in (
        _apply_cuda_tuning,
        _patch_dataloader,
        _patch_optimizer,
        _patch_ddp,
        _patch_tensorboard_images,
        _patch_ram_cache,
        _patch_async_save,
    ):
        try:
            step()
        except Exception:
            pass
`;
}

export function generateKaggleNotebookJson(
  config: ForkConfig,
  project: VoiceModelProject,
  downloads: DownloadModelItem[]
): string {
  const repoTarget = config.customRepoUrl.trim() || 'https://github.com/IAHispano/Applio.git';
  const themeB64 = toBase64Utf8(generateThemePython());
  const appB64 = toBase64Utf8(generateAppPython(config, project, downloads));
  const turboEnabled = config.turboTraining ?? true;
  const turboB64 = toBase64Utf8(generateTurboTrainerPython());

  const installLines: string[] = [
    `import base64`,
    `import ctypes.util`,
    `import hashlib`,
    `import os`,
    `import shutil`,
    `import subprocess`,
    `import sys`,
    `import urllib.request`,
    `from pathlib import Path`,
    `from IPython.display import clear_output`,
    ``,
    `repo_dir = Path("/kaggle/working/program_ml")`,
    `repo_url = "${repoTarget}"`,
    ``,
    `subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)`,
    `if repo_dir.exists() and (repo_dir / ".git").is_dir():`,
    `    subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=False)`,
    `else:`,
    `    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True)`,
    ``,
    `if shutil.which("fuser") is None or ctypes.util.find_library("portaudio") is None:`,
    `    subprocess.run(["apt", "update", "-y"], check=True)`,
    `    subprocess.run(["apt", "install", "-y", "portaudio19-dev", "psmisc"], check=True)`,
    ``,
    `uv_bin = shutil.which("uv") or str(Path(sys.executable).with_name("uv"))`,
  ];

  if (config.useUvVenv) {
    installLines.push(
      `venv_python = Path("/kaggle/.venv/bin/python")`,
      `if not venv_python.exists():`,
      `    subprocess.run([uv_bin, "venv", "/kaggle/.venv", "--python", "3.12", "--seed"], check=True)`,
      `py_exec = str(venv_python)`,
      `subprocess.run([uv_bin, "pip", "install", "-q", "-r", str(repo_dir / "requirements.txt"), "--extra-index-url", "https://download.pytorch.org/whl/cu128", "--index-strategy", "unsafe-best-match", "--python", py_exec], check=True)`,
      `subprocess.run([uv_bin, "pip", "install", "-q", "streamlit>=1.40.0", "matplotlib-inline", "websockets", "--python", py_exec], check=True)`
    );
  } else {
    installLines.push(
      `py_exec = sys.executable`,
      `subprocess.run([uv_bin, "pip", "install", "-q", "-r", str(repo_dir / "requirements.txt"), "--extra-index-url", "https://download.pytorch.org/whl/cu128", "--index-strategy", "unsafe-best-match", "--system"], check=True)`,
      `subprocess.run([uv_bin, "pip", "install", "-q", "streamlit>=1.40.0", "matplotlib-inline", "websockets", "--system"], check=True)`
    );
  }

  if (config.verifyOnnxGpu) {
    installLines.push(
      `subprocess.run([uv_bin, "pip", "install", "-q", "onnxruntime-gpu", "--python", py_exec] if ${config.useUvVenv ? 'True' : 'False'} else [uv_bin, "pip", "install", "-q", "onnxruntime-gpu", "--system"], check=False)`,
      `subprocess.run([py_exec, "-c", "import onnxruntime as ort; print('ONNX Providers:', ort.get_available_providers())"], check=False)`
    );
  }

  installLines.push(
    ``,
    `# Reset previously patched config so re-runs stay clean`,
    `if (repo_dir / ".git").is_dir():`,
    `    subprocess.run(["git", "-C", str(repo_dir), "checkout", "--", "assets/config_template.json"], check=False)`,
    ``,
    `# Enforce ${BRAND_NAME} engine defaults (fp32 precision, Discord presence off)`,
    `import json as _cfg_json`,
    `for _cfg_rel in ("assets/config_template.json", "assets/config.json"):`,
    `    _cfg_p = repo_dir / _cfg_rel`,
    `    if _cfg_p.is_file():`,
    `        try:`,
    `            _cfg_d = _cfg_json.loads(_cfg_p.read_text(encoding="utf-8"))`,
    `            _cfg_d["precision"] = "fp32"`,
    `            _cfg_d["discord_presence"] = False`,
    `            _cfg_p.write_text(_cfg_json.dumps(_cfg_d, indent=4, ensure_ascii=False), encoding="utf-8")`,
    `        except Exception:`,
    `            pass`,
    ``,
    `# Write ${BRAND_NAME} Streamlit WebUI and locked crimson theme`,
    `(repo_dir / ".streamlit").mkdir(parents=True, exist_ok=True)`,
    `(repo_dir / ".streamlit" / "config.toml").write_bytes(base64.b64decode("${themeB64}"))`,
    `(repo_dir / "streamlit_app.py").write_bytes(base64.b64decode("${appB64}"))`,
    ...(turboEnabled
      ? [
          `(repo_dir / "noregrets_turbo.py").write_bytes(base64.b64decode("${turboB64}"))`,
          `train_py = repo_dir / "rvc" / "train" / "train.py"`,
          `if train_py.is_file():`,
          `    _src = train_py.read_text(encoding="utf-8", errors="replace")`,
          `    if "noregrets_turbo.apply()" not in _src:`,
          `        _lines = _src.splitlines(keepends=True)`,
          `        _ins = 0`,
          `        for _i, _ln in enumerate(_lines[:40]):`,
          `            if _ln.startswith("from __future__ import"):`,
          `                _ins = _i + 1`,
          `        _hook = "import os as _nr_os, sys as _nr_sys\\n_nr_sys.path.insert(0, _nr_os.getcwd())\\ntry:\\n    import noregrets_turbo\\n    noregrets_turbo.apply()\\nexcept Exception:\\n    pass\\n"`,
          `        _lines.insert(_ins, _hook)`,
          `        train_py.write_text("".join(_lines), encoding="utf-8")`,
        ]
      : []),
    ``,
    `# Best-effort prefetch of the 6 Free Samples WAVs from ${REPO_URL}`,
    `fs_dir = repo_dir / "assets" / "free_samples"`,
    `au_dir = repo_dir / "assets" / "audios"`,
    `fs_dir.mkdir(parents=True, exist_ok=True)`,
    `au_dir.mkdir(parents=True, exist_ok=True)`,
    `for _wav_name in (`,
    `    "MaleSpeakingTest-1_931b5ffd.wav",`,
    `    "MaleSpeakingTest-2_c2e6e7fe.wav",`,
    `    "MaleSpeakingTest-3_d78e8233.wav",`,
    `    "FemaleSpeakingTest-1_bb3d674f.wav",`,
    `    "FemaleSpeakingTest-2_d1823132.wav",`,
    `    "FemaleSpeakingTest-3_458bd4fc.wav",`,
    `):`,
    `    _dst_fs = fs_dir / _wav_name`,
    `    if not _dst_fs.is_file():`,
    `        try:`,
    `            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/HunterThatsMe/NoRegretsRVC/main/public/assets/aistudio/{_wav_name}", _dst_fs)`,
    `            shutil.copy2(_dst_fs, au_dir / _wav_name)`,
    `        except Exception:`,
    `            pass`,
    ``,
    `os.chdir(repo_dir)`,
    `prereq_cmd = [py_exec, "core.py", "prerequisites", "--models", "--exe", "--pretraineds-hifigan"]`,
    `subprocess.run(prereq_cmd, check=True)`
  );

  const bootItems: { name: string; url: string; category: string }[] = downloads.map((d) => ({
    name: d.name,
    url: d.url,
    category: d.category,
  }));
  if (project.customPretrained) {
    const autoD = !project.customPretrainedD.trim()
      ? inferMatchingDUrl(project.customPretrainedG)
      : '';
    for (const rawUrl of [project.customPretrainedG, project.customPretrainedD || autoD]) {
      const clean = (rawUrl || '').trim();
      if (/^https?:\/\//i.test(clean) && !bootItems.some((b) => b.url === clean)) {
        bootItems.push({
          name: clean.split('/').pop()?.split('?')[0] || 'custom_pretrained.pth',
          url: clean,
          category: 'pretrained',
        });
      }
    }
  }

  if (bootItems.length > 0) {
    installLines.push(
      ``,
      `# Pre-boot Model & Custom Pretrains Downloader (HF repo auto-discovery, ZIP & G/D .pth)`,
      `import json as _json`,
      `import urllib.parse`,
      `import zipfile as _zipfile`,
      `boot_downloads = ${JSON.stringify(bootItems, null, 4)}`,
      `dl_dir = repo_dir / "logs" / "downloads"`,
      `custom_pretrain_dir = repo_dir / "rvc" / "models" / "pretraineds" / "custom"`,
      `dl_dir.mkdir(parents=True, exist_ok=True)`,
      `custom_pretrain_dir.mkdir(parents=True, exist_ok=True)`,
      `def _download_hf_repo_pretrains(_repo_url, _dst_dir):`,
      `    _parts = [p for p in urllib.parse.urlsplit(_repo_url).path.strip("/").split("/") if p]`,
      `    if len(_parts) < 2:`,
      `        return`,
      `    _owner_repo = f"{_parts[0]}/{_parts[1]}"`,
      `    _api = f"https://huggingface.co/api/models/{_owner_repo}/tree/main"`,
      `    with urllib.request.urlopen(_api, timeout=30) as _resp:`,
      `        _tree = _json.loads(_resp.read().decode("utf-8"))`,
      `    for _node in _tree:`,
      `        _fpath = _node.get("path", "")`,
      `        if _fpath.lower().endswith(".pth"):`,
      `            _fname = Path(_fpath).name`,
      `            _out = _dst_dir / _fname`,
      `            if not _out.is_file():`,
      `                _file_url = f"https://huggingface.co/{_owner_repo}/resolve/main/{urllib.parse.quote(_fpath)}?download=true"`,
      `                print("Downloading discovered pretrain", _fname, "->", _out)`,
      `                urllib.request.urlretrieve(_file_url, _out)`,
      `for entry in boot_downloads:`,
      `    _url = entry["url"].strip()`,
      `    _cat = entry.get("category", "voice")`,
      `    _target_dir = custom_pretrain_dir if _cat == "pretrained" else dl_dir`,
      `    _parsed_path = urllib.parse.urlsplit(_url).path`,
      `    _raw_name = urllib.parse.unquote(_parsed_path.split("/")[-1]) or entry["name"]`,
      `    try:`,
      `        if _cat == "pretrained" and "huggingface.co/" in _url.lower() and "/resolve/" not in _url and not _parsed_path.lower().endswith((".pth", ".zip")):`,
      `            _download_hf_repo_pretrains(_url, _target_dir)`,
      `        elif _cat == "pretrained" and _parsed_path.lower().endswith(".zip"):`,
      `            _zip_tmp = _target_dir / (_raw_name if _raw_name.endswith(".zip") else "custom_pretrains.zip")`,
      `            print("Downloading pretrained ZIP", _url, "->", _zip_tmp)`,
      `            urllib.request.urlretrieve(_url, _zip_tmp)`,
      `            with _zipfile.ZipFile(_zip_tmp, "r") as _zf:`,
      `                for _member in _zf.namelist():`,
      `                    if not _member.endswith("/") and Path(_member).name and _member.lower().endswith(".pth"):`,
      `                        _out_pth = _target_dir / Path(_member).name`,
      `                        _out_pth.write_bytes(_zf.read(_member))`,
      `                        print("Extracted", _out_pth.name)`,
      `            _zip_tmp.unlink(missing_ok=True)`,
      `        else:`,
      `            _dst = _target_dir / _raw_name`,
      `            if not _dst.is_file():`,
      `                print("Downloading", entry["name"], "->", _dst)`,
      `                urllib.request.urlretrieve(_url, _dst)`,
      `    except Exception as _dl_err:`,
      `        print("Skipping unreachable download", _url, ":", _dl_err)`
    );
  }

  installLines.push(
    ``,
    `cloudflared_path = repo_dir.parent / "cloudflared"`,
    `cloudflared_url = "https://github.com/cloudflare/cloudflared/releases/download/2026.9.0/cloudflared-linux-amd64"`,
    `cloudflared_sha256 = "53b7a7a5420d188758d24341294acb0d1bca54296548ac05e38811a694ac6134"`,
    `if not (cloudflared_path.is_file() and hashlib.sha256(cloudflared_path.read_bytes()).hexdigest() == cloudflared_sha256):`,
    `    tmp = cloudflared_path.with_suffix(".download")`,
    `    with urllib.request.urlopen(cloudflared_url, timeout=120) as resp, open(tmp, "wb") as dst:`,
    `        shutil.copyfileobj(resp, dst)`,
    `    if hashlib.sha256(tmp.read_bytes()).hexdigest() != cloudflared_sha256:`,
    `        tmp.unlink(missing_ok=True)`,
    `        raise RuntimeError("cloudflared checksum verification failed.")`,
    `    os.replace(tmp, cloudflared_path)`,
    `cloudflared_path.chmod(0o755)`,
    `clear_output()`,
    `print("${BRAND_NAME} Streamlit setup complete! Model: ${sanitizeModelName(project.name)} (${project.samplingRate} · ${project.vocoder})")`
  );

  const pyExecutableExpr = config.useUvVenv ? '"/kaggle/.venv/bin/python"' : 'sys.executable';

  const startLines: string[] = [
    `import os`,
    `import re`,
    `import secrets`,
    `import subprocess`,
    `import sys`,
    `import time`,
    `from collections import deque`,
    `from pathlib import Path`,
    `from IPython.display import clear_output`,
    ``,
    `Tunnel = "${config.tunnelMode}" #@param ["Gradio + Cloudflare", "Horizon"]`,
    `horizon_id = "${config.horizonId}" #@param {type:"string"}`,
    ``,
    `repo_dir = Path("/kaggle/working/program_ml")`,
    `cloudflared_path = repo_dir.parent / "cloudflared"`,
    `py_exec = ${pyExecutableExpr}`,
    ``,
    `app_user = "noregrets"`,
    `app_pass = ${config.randomAuthEnabled ? 'secrets.token_urlsafe(24)' : '""'}`,
    ``,
    `cf_pattern = r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com"`,
    `hz_pattern = r"https://[a-zA-Z0-9.-]+\\.hrzn\\.run"`,
    `tunnel_procs = []`,
    `tunnel_logs = []`,
    `app_proc = None`,
    ``,
    `def start_tunnel(cmd, name, pattern):`,
    `    out_path = repo_dir / f"{name}.log"`,
    `    f = open(out_path, "w", encoding="utf-8")`,
    `    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)`,
    `    tunnel_procs.append(proc)`,
    `    tunnel_logs.append(f)`,
    `    deadline = time.time() + 60`,
    `    while time.time() < deadline:`,
    `        txt = out_path.read_text(encoding="utf-8", errors="replace") if out_path.exists() else ""`,
    `        m = re.search(pattern, txt)`,
    `        if m:`,
    `            return m.group(0)`,
    `        time.sleep(1)`,
    `    raise RuntimeError(f"Tunnel {name} failed to emit URL")`,
    ``,
    `os.chdir(repo_dir)`,
    `subprocess.run(["fuser", "-k", "6969/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)`,
    ``,
    `if Tunnel == "Horizon":`,
    `    if not horizon_id.strip():`,
    `        raise ValueError("Enter your horizon_id before running Horizon tunnel.")`,
    `    subprocess.run(["npm", "install", "-g", "@hrzn/cli"], check=True)`,
    `    subprocess.run(["hrzn", "login", horizon_id], check=True)`,
    ``,
    `# --- ${BRAND_NAME} Kaggle Live Monitor (spaced rows + mid-epoch + Speed|Step|Epoch table) ---`,
    `BOLD = "\\033[1m"`,
    `RESET = "\\033[0m"`,
    `ANSI_RE = re.compile(r"\\x1B\\[[0-?]*[ -/]*[@-~]")`,
    `EPOCH_RE = re.compile(r"epoch=(\\d+)\\s*\\|\\s*step=(\\d+).*?training_speed=(\\d+:\\d{2}:\\d{2})(?:.*?lowest_value=([0-9.]+))?")`,
    `TQDM_RE = re.compile(r"(\\d+)%\\|[^|]*\\|\\s*(\\d+)/(\\d+)\\s*\\[([^,\\]]+),\\s*([0-9.]+\\s*(?:it/s|s/it))")`,
    `LOCAL_URL_RE = re.compile(r"(?:Running on local URL|Local URL|Network URL):\\s*(https?://\\S+)")`,
    `NOISE_RE = re.compile(r"(FutureWarning|UserWarning|DeprecationWarning|RuntimeWarning|pkg_resources|warnings\\.warn|oneDNN|tensorflow/core|computation_placer|numa_node|Successfully opened dynamic library|weights_only|HTTP Request:|Checking out|libpng warning|ALSA|jack server|You can now view your Streamlit)", re.IGNORECASE)`,
    `MILESTONE_RE = re.compile(r"(Starting preprocess|Preprocess completed|preprocessed successfully|Starting pitch extraction|Pitch extraction completed|Starting embedding extraction|Embedding extraction completed|Saved index file|Starting training|Saving checkpoint|Saving model|Training finished|Training completed|Index file.*generated|Using .* vocoder|Using .* for training)", re.IGNORECASE)`,
    `ERROR_START_RE = re.compile(r"(Traceback \\(most recent call last\\)|RuntimeError:|ValueError:|FileNotFoundError:|OSError:|ModuleNotFoundError:|ImportError:|CUDA out of memory|OutOfMemoryError|Exception:|\\bERROR:)")`,
    ``,
    `session_links = []`,
    `milestones = deque(maxlen=6)`,
    `errors = deque(maxlen=25)`,
    `epoch_rows = deque(maxlen=15)`,
    `seen_epochs = set()`,
    `live_mid_epoch = ""`,
    `last_tq_render = 0.0`,
    `tb_lines_seen = 0`,
    ``,
    `def upsert_link(label, value):`,
    `    for idx, (k, _) in enumerate(session_links):`,
    `        if k.strip() == label.strip():`,
    `            session_links[idx] = (label, value)`,
    `            return`,
    `    session_links.append((label, value))`,
    ``,
    `def render_monitor(status="Running"):`,
    `    w_speed = max(9, max((len(r[0]) for r in epoch_rows), default=0))`,
    `    w_step = max(8, max((len(r[1]) for r in epoch_rows), default=0))`,
    `    w_epoch = max(6, max((len(r[2]) for r in epoch_rows), default=0))`,
    `    lines = [`,
    `        f"{BOLD}${BRAND_NAME} Live Monitor (Beta 0.0.0){RESET}  [{status}]",`,
    `        "=" * 64,`,
    `        "",`,
    `    ]`,
    `    if app_pass:`,
    `        lines.append(f"WebUI Login      : {app_user} / {app_pass}")`,
    `        lines.append("")`,
    `    for label, val in session_links:`,
    `        if val:`,
    `            lines.append(f"{label:<17}: {val}")`,
    `            lines.append("")`,
    `    lines.append("-" * 64)`,
    `    lines.append("")`,
    `    if milestones:`,
    `        for m in milestones:`,
    `            lines.append(f"[Stage] {m}")`,
    `        lines.append("")`,
    `    if live_mid_epoch:`,
    `        lines.append(f"{BOLD}Live Epoch:{RESET} {live_mid_epoch}")`,
    `        lines.append("")`,
    `    if epoch_rows:`,
    `        lines.append(f"{'Speed':<{w_speed}} | {'Step':<{w_step}} | {BOLD}{'Epoch':<{w_epoch}}{RESET} | Lowest Loss")`,
    `        lines.append(f"{'-' * w_speed}-+-{'-' * w_step}-+-{'-' * w_epoch}-+------------")`,
    `        for spd, stp, ep, low in epoch_rows:`,
    `            lines.append(f"{spd:<{w_speed}} | {stp:<{w_step}} | {BOLD}{ep:<{w_epoch}}{RESET} | {low}")`,
    `    elif not errors:`,
    `        lines.append("Waiting for training epochs...")`,
    `    if errors:`,
    `        lines.append("")`,
    `        lines.append(f"{BOLD}Errors:{RESET}")`,
    `        for err in errors:`,
    `            lines.append(f"  {err}")`,
    `    clear_output(wait=True)`,
    `    print("\\n".join(lines), flush=True)`,
    ``,
    `try:`,
    `    env = os.environ.copy()`,
    `    if app_pass:`,
    `        env["APPLIO_AUTH_USERNAME"] = app_user`,
    `        env["APPLIO_AUTH_PASSWORD"] = app_pass`,
    `    env.update({`,
    `        "PYTHONWARNINGS": "ignore",`,
    `        "PYTHONUNBUFFERED": "1",`,
    `        "TF_CPP_MIN_LOG_LEVEL": "3",`,
    `        "TORCH_CPP_LOG_LEVEL": "ERROR",`,
    `        "TOKENIZERS_PARALLELISM": "false",`,
    `    })`,
    ``,
    `    if Tunnel == "Gradio + Cloudflare":`,
    `        cf_url = start_tunnel([str(cloudflared_path), "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", "http://127.0.0.1:6969"], "webui-cf", cf_pattern)`,
    `        upsert_link("${BRAND_NAME} URL", cf_url)`,
    `        upsert_link("Cloudflare URL", cf_url)`,
    `    else:`,
    `        web_url = start_tunnel(["hrzn", "tunnel", "http://localhost:6969"], "webui-hrzn", hz_pattern)`,
    `        upsert_link("${BRAND_NAME} URL", web_url)`,
    ``,
    `    app_cmd = [`,
    `        py_exec, "-u", "-m", "streamlit", "run", "streamlit_app.py",`,
    `        "--server.address", "0.0.0.0",`,
    `        "--server.port", "6969",`,
    `        "--server.headless", "true",`,
    `        "--server.enableCORS", "false",`,
    `        "--server.enableXsrfProtection", "false",`,
    `    ]`,
    `    render_monitor("Starting ${BRAND_NAME} WebUI")`,
    `    app_proc = subprocess.Popen(app_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)`,
    `    assert app_proc.stdout is not None`,
    `    for raw_line in app_proc.stdout:`,
    `        line = ANSI_RE.sub("", raw_line).replace("\\r", "").strip()`,
    `        if not line:`,
    `            continue`,
    `        loc_m = LOCAL_URL_RE.search(line)`,
    `        if loc_m:`,
    `            upsert_link("Local UI URL", loc_m.group(1))`,
    `            render_monitor("Ready")`,
    `            continue`,
    `        ep_m = EPOCH_RE.search(line)`,
    `        if ep_m:`,
    `            ep_num, stp_num, spd_str = ep_m.group(1), ep_m.group(2), ep_m.group(3)`,
    `            low_str = ep_m.group(4) or "-"`,
    `            key = (ep_num, stp_num)`,
    `            if key not in seen_epochs:`,
    `                seen_epochs.add(key)`,
    `                epoch_rows.append((spd_str, stp_num, ep_num, low_str))`,
    `                live_mid_epoch = ""`,
    `                render_monitor("Training")`,
    `            continue`,
    `        tq_m = TQDM_RE.search(line)`,
    `        if tq_m:`,
    `            now_ts = time.time()`,
    `            if now_ts - last_tq_render >= 2.0 or tq_m.group(1) == "100":`,
    `                last_tq_render = now_ts`,
    `                live_mid_epoch = f"{tq_m.group(1)}% ({tq_m.group(2)}/{tq_m.group(3)}) · {tq_m.group(4)} · {tq_m.group(5)}"`,
    `                render_monitor("Training")`,
    `            continue`,
    `        if NOISE_RE.search(line) or "%|" in line or "it/s]" in line or "s/it]" in line:`,
    `            continue`,
    `        if tb_lines_seen > 0:`,
    `            errors.append(line)`,
    `            tb_lines_seen = 0 if re.match(r"^[A-Za-z_]*(Error|Exception):", line) else (tb_lines_seen - 1)`,
    `            render_monitor("Error")`,
    `            continue`,
    `        if ERROR_START_RE.search(line):`,
    `            errors.append(line)`,
    `            if "Traceback" in line:`,
    `                tb_lines_seen = 25`,
    `            render_monitor("Error")`,
    `            continue`,
    `        if MILESTONE_RE.search(line):`,
    `            if not milestones or milestones[-1] != line:`,
    `                milestones.append(line)`,
    `                render_monitor("Running")`,
    `    app_proc.wait()`,
    `finally:`,
    `    for p in reversed(tunnel_procs):`,
    `        if p.poll() is None: p.terminate()`,
    `    if app_proc is not None and app_proc.poll() is None:`,
    `        app_proc.terminate()`,
    `    for f in tunnel_logs:`,
    `        f.close()`,
    `    subprocess.run(["fuser", "-k", "6969/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)`,
  ];

  const notebook = {
    nbformat: 4,
    nbformat_minor: 4,
    metadata: {
      kernelspec: {
        display_name: 'Python 3',
        language: 'python',
        name: 'python3',
      },
      language_info: {
        name: 'python',
        version: '3.12.0',
      },
      kaggle: {
        accelerator: 'nvidiaTeslaT4',
        isGpuEnabled: true,
        isInternetEnabled: true,
        language: 'python',
        sourceType: 'notebook',
      },
    },
    cells: [
      {
        cell_type: 'markdown',
        metadata: {},
        source: [
          `## **${BRAND_NAME}**\n`,
          `Custom RVC voice model training environment built on Applio + Streamlit (${project.samplingRate} · ${project.vocoder} · ${project.pitchExtractor.toUpperCase()}).\n\n`,
          `- **Dataset Path**: \`${project.datasetPath || '/kaggle/input/datasets/your-username/dataset'}\`\n`,
          `- **Included Tabs**: ${Object.entries(config.activeTabs).filter(([k, v]) => v && k !== 'extra' && k !== 'tensorboard').map(([k]) => k).join(', ')}\n`,
          `- **Tunnel & Auth**: ${config.tunnelMode} with ${config.randomAuthEnabled ? 'auto-generated 24-byte WebUI password' : 'standard access'}\n`,
        ],
      },
      {
        cell_type: 'markdown',
        metadata: {},
        source: ['## 1. Install Dependencies & Apply NoRegretsRVC Streamlit WebUI\n'],
      },
      {
        cell_type: 'code',
        execution_count: null,
        metadata: { trusted: true },
        outputs: [],
        source: installLines.map((l) => l + '\n'),
      },
      {
        cell_type: 'markdown',
        metadata: {},
        source: [
          '## 2. Launch Streamlit WebUI & Live Monitor\n',
        ],
      },
      {
        cell_type: 'code',
        execution_count: null,
        metadata: { trusted: true },
        outputs: [],
        source: startLines.map((l) => l + '\n'),
      },
    ],
  };

  return JSON.stringify(notebook, null, 2);
}

function toBase64Utf8(str: string): string {
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(str, 'utf8').toString('base64');
  }
  const bytes = new TextEncoder().encode(str);
  let bin = '';
  for (let i = 0; i < bytes.length; i++) {
    bin += String.fromCharCode(bytes[i]);
  }
  return btoa(bin);
}

export function generateForkBundle(
  config: ForkConfig,
  project: VoiceModelProject,
  downloads: DownloadModelItem[]
): GeneratedBundleFile[] {
  return [
    {
      id: 'notebook',
      filename: `${BRAND_NAME}_Kaggle.ipynb`,
      zipPath: `assets/${BRAND_NAME}_Kaggle.ipynb`,
      language: 'json',
      description:
        'Kaggle-ready notebook launching the crimson NoRegretsRVC Streamlit WebUI with SHA-256 verified tunnels, randomized password, and dual-T4 Turbo setup.',
      content: generateKaggleNotebookJson(config, project, downloads),
    },
    {
      id: 'app-py',
      filename: 'streamlit_app.py',
      zipPath: 'streamlit_app.py',
      language: 'python',
      description:
        'NoRegretsRVC Streamlit WebUI registering Training, Models, Samples, Free Samples, Diagnostics, Voice Blender, TTS, and Settings.',
      content: generateAppPython(config, project, downloads),
    },
    {
      id: 'theme-py',
      filename: 'config.toml',
      zipPath: '.streamlit/config.toml',
      language: 'json',
      description:
        'Fixed crimson-red Streamlit theme configuration built specifically for NoRegretsRVC.',
      content: generateThemePython(),
    },
    {
      id: 'turbo-py',
      filename: 'noregrets_turbo.py',
      zipPath: 'noregrets_turbo.py',
      language: 'python',
      description:
        'Lossless Kaggle training speedup patch (adaptive RAM cache, fused AdamW, DataLoader tuning, cuDNN benchmark, DDP bucket view, async .pth writes).',
      content: generateTurboTrainerPython(),
    },
    {
      id: 'commands-sh',
      filename: 'run_commands.sh',
      zipPath: 'run_commands.sh',
      language: 'bash',
      description: `Generated Kaggle shell commands for "${sanitizeModelName(project.name)}" (${project.samplingRate} · ${project.vocoder}).`,
      content: `#!/usr/bin/env bash\nset -euo pipefail\n\n${buildTrainPipelineCommands(project)}\n`,
    },
  ];
}
