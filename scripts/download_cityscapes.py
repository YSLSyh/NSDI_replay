"""通过用户自己的 Cityscapes 授权账号在远程服务器下载两份官方 ZIP。"""

from __future__ import annotations

import getpass
import hashlib
import os
import sys
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "raw" / "cityscapes" / "downloads"
LOG = ROOT / "artifacts" / "cityscapes_download.log"
BASE = "https://www.cityscapes-dataset.com"
PACKAGES = (
    (1, "gtFine_trainvaltest.zip", "4237c19de34c8a376e9ba46b495d6f66"),
    (3, "leftImg8bit_trainvaltest.zip", "0a6e97e94b616a514066c9e2adb0c97f"),
)
CHUNK = 4 * 1024 * 1024


def digest(path: Path) -> str:
    checksum = hashlib.md5()  # 官方仅发布 MD5；这里用于与官网文件校验值比较。
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def download_one(session: requests.Session, package_id: int, name: str, expected_md5: str) -> None:
    final = DESTINATION / name
    partial = DESTINATION / (name + ".part")
    if final.exists():
        if digest(final) != expected_md5:
            raise ValueError(f"已有文件与官网 MD5 不一致，未覆盖：{final}")
        print(f"已校验：{name}", flush=True)
        return
    url = f"{BASE}/file-handling/?packageID={package_id}"
    attempts = 0
    while attempts < 30:
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise RuntimeError("服务器返回 HTML，Cityscapes 登录会话可能已失效")
                if offset and response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError(f"续传范围不匹配：{content_range}")
                    mode = "ab"
                elif offset and response.status_code == 200:
                    # 服务器不支持 Range 时，从头写入本任务自己的 .part 文件。
                    print(f"{name} 不支持续传，从头下载", flush=True)
                    mode = "wb"
                    offset = 0
                elif response.status_code == 200:
                    mode = "wb"
                else:
                    raise RuntimeError(f"意外的 HTTP 状态：{response.status_code}")
                with partial.open(mode) as target:
                    first = True
                    for chunk in response.iter_content(chunk_size=CHUNK):
                        if not chunk:
                            continue
                        if first and offset == 0 and not chunk.startswith(b"PK"):
                            raise RuntimeError("下载内容不是 ZIP，已停止写入")
                        first = False
                        target.write(chunk)
                actual = digest(partial)
                if actual != expected_md5:
                    raise ValueError(f"{name} 的 MD5 不匹配：{actual}，期望 {expected_md5}")
                os.replace(partial, final)
                print(f"已完成并校验：{final} ({final.stat().st_size} 字节)", flush=True)
                return
        except (requests.RequestException, OSError) as error:
            attempts += 1
            print(f"{name} 下载中断，第 {attempts}/30 次重试：{error}", flush=True)
            time.sleep(min(60, attempts * 5))
    raise RuntimeError(f"{name} 重试 30 次仍未完成，保留 .part 以供续传")


def main() -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("请在远程服务器交互式终端运行，以隐藏输入密码")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    print("Cityscapes 官方账号仅用于本次登录；密码与会话 Cookie 不写入文件。")
    username = input("Cityscapes 用户名或邮箱：").strip()
    password = getpass.getpass("Cityscapes 密码（输入不可见）：")
    if not username or not password:
        raise ValueError("用户名和密码不能为空")
    session = requests.Session()
    session.get(f"{BASE}/login/", timeout=30).raise_for_status()
    response = session.post(
        f"{BASE}/login/",
        data={"username": username, "password": password, "submit": "Login"},
        timeout=30,
    )
    response.raise_for_status()
    check = session.get(f"{BASE}/downloads/", timeout=30)
    check.raise_for_status()
    if "Logout" not in check.text or "gtFine_trainvaltest.zip" not in check.text:
        raise RuntimeError("官网未确认登录成功；请核对账号是否已经激活")
    cookies = session.cookies.copy()
    session.close()
    del password
    pid = os.fork()
    if pid:
        print(f"登录成功；下载已转入后台，PID={pid}，日志：{LOG}")
        return
    os.setsid()
    sys.stdin.close()
    output = LOG.open("a", buffering=1, encoding="utf-8")
    os.dup2(output.fileno(), 1)
    os.dup2(output.fileno(), 2)
    print(f"开始下载：{time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    background = requests.Session()
    background.cookies.update(cookies)
    try:
        for package in PACKAGES:
            download_one(background, *package)
        print("Cityscapes 两个官方包均已完成并通过官网 MD5 校验", flush=True)
    except BaseException as error:
        print(f"Cityscapes 下载失败：{type(error).__name__}: {error}", flush=True)
        raise
    finally:
        background.close()
    os._exit(0)


if __name__ == "__main__":
    main()
