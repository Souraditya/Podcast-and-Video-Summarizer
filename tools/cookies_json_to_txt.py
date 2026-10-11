"""
Convert a browser-extension JSON cookie export into the Netscape cookies.txt
yt-dlp expects -- keeping ONLY the youtube.com family and the bare .google.com
domain, so nothing else from your browser (Gmail, Drive, other sites) ends
up in the file.

Usage:
    python tools/cookies_json_to_txt.py cookies.json youtube_cookies.txt
"""
import json
import sys

EXACT = {".google.com", "google.com", "www.google.com"}


def main(src: str, dst: str) -> None:
    cookies = json.load(open(src, encoding="utf-8"))
    lines = ["# Netscape HTTP Cookie File"]
    kept = 0
    for c in cookies:
        domain = c.get("domain", "")
        bare = domain.lstrip(".")
        if not (bare == "youtube.com" or bare.endswith(".youtube.com") or domain in EXACT):
            continue
        expires = int(c.get("expirationDate", 0) or 0)
        lines.append("\t".join([
            domain, "TRUE" if domain.startswith(".") else "FALSE", c.get("path", "/"),
            "TRUE" if c.get("secure") else "FALSE", str(expires), c["name"], c["value"],
        ]))
        kept += 1
    open(dst, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print(f"kept {kept} youtube + google.com cookies -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
