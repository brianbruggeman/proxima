"""Python client hitting a proxima-fronted hello-world endpoint.

usage:
    # in one shell, start the server:
    proxima serve --config ../../../config/01-hello-rest/proxima.toml \
        --mount '/{path*}' --addr 127.0.0.1:8080

    # in another:
    python client.py

Set PROXIMA_HELLO_URL to point at a non-default bind address.
"""

import os
import sys
import urllib.request


def main() -> int:
    url = os.environ.get("PROXIMA_HELLO_URL", "http://127.0.0.1:8080/anything")
    response = urllib.request.urlopen(url, timeout=5)
    body = response.read().decode("utf-8")
    print(body, end="")
    return 0 if body.strip() == "hello world" else 1


if __name__ == "__main__":
    sys.exit(main())
