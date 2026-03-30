"""
download.py — One-stop data download for QANet / Assignment 1.

Usage (from Assignment1.ipynb):
    from Tools.download import download
    download()                          # uses default _data/ layout
    download(data_dir="my_data")        # custom root

Individual steps:
    from Tools.download import download_squad, download_glove, download_spacy_model
    download_squad()
    download_glove()
    download_spacy_model()
"""

import os
import ssl
import subprocess
import sys
import urllib.request
import zipfile
from urllib.error import URLError


# ---------------------------------------------------------------------------
# Mini dataset — fill in after uploading mini_data.zip to GitHub Releases
# ---------------------------------------------------------------------------

MINI_DATA_URL = (
    "https://github.com/usyddeeplearning/Assignment1--2026/releases/download/data-v1.00/mini_data.zip"
)

_SSL_OPENER_CONFIGURED = False


def _ensure_ssl_opener() -> None:
    """Configure urllib to use certifi's CA bundle when available."""
    global _SSL_OPENER_CONFIGURED
    if _SSL_OPENER_CONFIGURED:
        return

    try:
        import certifi
    except ImportError:
        _SSL_OPENER_CONFIGURED = True
        return

    context = ssl.create_default_context(cafile=certifi.where())
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
    urllib.request.install_opener(opener)
    _SSL_OPENER_CONFIGURED = True


def _urlretrieve(url: str, dest: str, reporthook=None) -> None:
    """Wrapper around urllib.request.urlretrieve with actionable SSL errors."""
    try:
        urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    except URLError as exc:
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            raise RuntimeError(
                "SSL certificate verification failed while downloading data. "
                "On macOS, install certificates and retry: "
                "open '/Applications/Python 3.x/Install Certificates.command' "
                "(matching your Python version). "
                "If using a virtual environment, also install certifi: pip install certifi."
            ) from exc
        raise


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _download_file(url: str, dest: str) -> None:
    """Download *url* to *dest*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    _ensure_ssl_opener()
    try:
        from tqdm import tqdm

        class _Hook:
            def __init__(self):
                self._t = None

            def __call__(self, n_blocks, block_size, total):
                if self._t is None:
                    self._t = tqdm(
                        total=total, unit="B", unit_scale=True,
                        desc=os.path.basename(dest), leave=True,
                    )
                downloaded = n_blocks * block_size
                self._t.update(downloaded - self._t.n)

            def close(self):
                if self._t is not None:
                    self._t.close()

        hook = _Hook()
        _urlretrieve(url, dest, reporthook=hook)
        hook.close()

    except ImportError:
        print(f"Downloading {os.path.basename(dest)} …")
        _urlretrieve(url, dest)
        print(f"  Saved → {dest}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_squad(squad_dir: str = os.path.join("_data", "squad")) -> None:
    """Download SQuAD v1.1 train and dev JSON files.

    Parameters
    ----------
    squad_dir:
        Directory to save the two JSON files.  Created if absent.
    """
    os.makedirs(squad_dir, exist_ok=True)
    base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset"
    for fname in ("train-v1.1.json", "dev-v1.1.json"):
        dest = os.path.join(squad_dir, fname)
        if os.path.exists(dest):
            print(f"  [skip] {dest} already exists.")
        else:
            _download_file(f"{base_url}/{fname}", dest)


def download_glove(glove_dir: str = os.path.join("_data", "glove")) -> None:
    """Download and unzip GloVe 840B 300-d word vectors.

    Parameters
    ----------
    glove_dir:
        Directory for the zip archive and extracted .txt file.
    """
    os.makedirs(glove_dir, exist_ok=True)
    txt_path = os.path.join(glove_dir, "glove.840B.300d.txt")
    zip_path = os.path.join(glove_dir, "glove.840B.300d.zip")

    if os.path.exists(txt_path):
        print(f"  [skip] {txt_path} already exists.")
        return

    if not os.path.exists(zip_path):
        _download_file(
            "http://nlp.stanford.edu/data/glove.840B.300d.zip",
            zip_path,
        )

    print(f"Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(glove_dir)
    print(f"  Extracted → {glove_dir}")


def download_mini(
    url: str = MINI_DATA_URL,
    data_dir: str = "_data",
) -> None:
    """Download and extract the pre-built mini dataset package.

    Downloads ``mini_data.zip`` from *url* (a GitHub Releases asset) and
    extracts it into *data_dir*, producing::

        _data/
          squad/
            train-mini.json      (~150 train articles)
            dev-v1.1.json        (full dev set, 48 articles)
          glove/
            glove.mini.txt       (GloVe vectors filtered to mini vocab)

    Also downloads the spaCy ``en`` model.

    Parameters
    ----------
    url:
        Direct download URL for ``mini_data.zip``.  Defaults to
        ``MINI_DATA_URL`` — update that constant after uploading to Releases.
    data_dir:
        Root directory for extracted data (default ``"_data"``).
    """
    print("=" * 60)
    print("Step 1 / 2  —  Mini dataset (SQuAD + GloVe)")
    print("=" * 60)

    squad_sentinel = os.path.join(data_dir, "squad", "train-mini.json")
    glove_sentinel = os.path.join(data_dir, "glove", "glove.mini.txt")

    if os.path.exists(squad_sentinel) and os.path.exists(glove_sentinel):
        print(f"  [skip] Mini dataset already present in {data_dir}/.")
    else:
        zip_dest = os.path.join(data_dir, "mini_data.zip")
        os.makedirs(data_dir, exist_ok=True)
        _download_file(url, zip_dest)

        print(f"Extracting {zip_dest} …")
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(data_dir)
        os.remove(zip_dest)
        print(f"  Extracted → {data_dir}/")

    print()
    print("=" * 60)
    print("Step 2 / 2  —  spaCy language model")
    print("=" * 60)
    download_spacy_model()

    print()
    print("Mini dataset download complete.")


def download_spacy_model(model: str = "en_core_web_sm") -> None:
    """Download the spaCy language model required for tokenisation.

    Parameters
    ----------
    model:
        spaCy model name (default ``"en"``).
    """
    print(f"Downloading spaCy model '{model}' …")
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", model],
        capture_output=True, text=True,
    )
    if result.returncode != 0 and model == "en":
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            capture_output=True, text=True,
        )
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            f"`spacy download {model}` failed (exit code {result.returncode})"
        )
    print(result.stdout.strip())


def download(data_dir: str = "_data") -> None:
    """Download all required data: SQuAD v1.1, GloVe 840B, and the spaCy model.

    Creates the following layout under *data_dir*::

        _data/
          squad/
            train-v1.1.json
            dev-v1.1.json
          glove/
            glove.840B.300d.zip
            glove.840B.300d.txt

    Parameters
    ----------
    data_dir:
        Root directory for raw data (default ``"_data"``).
    """
    print("=" * 60)
    print("Step 1 / 3  —  SQuAD v1.1")
    print("=" * 60)
    download_squad(os.path.join(data_dir, "squad"))

    print()
    print("=" * 60)
    print("Step 2 / 3  —  GloVe 840B 300d")
    print("=" * 60)
    download_glove(os.path.join(data_dir, "glove"))

    print()
    print("=" * 60)
    print("Step 3 / 3  —  spaCy language model")
    print("=" * 60)
    download_spacy_model()

    print()
    print("All downloads complete.")
