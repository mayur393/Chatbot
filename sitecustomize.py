"""
sitecustomize.py
----------------
Fallback DNS resolver hook using public DNS (8.8.8.8, 1.1.1.1) when the local router DNS fails.
Transparently ensures pip, requests, and streamlit resolve domains reliably.
"""

import socket

# Force IPv4 in urllib3 to avoid Windows IPv6 routing drops / WinError 10054
try:
    import urllib3.util.connection as _urllib_conn
    _urllib_conn.allowed_gai_family = lambda: socket.AF_INET
except Exception:
    pass

_orig_getaddrinfo = socket.getaddrinfo


def _query_public_dns(domain, dns_ip="8.8.8.8"):
    try:
        packet = bytearray(b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00")
        for part in domain.split("."):
            packet.append(len(part))
            packet.extend(part.encode("utf-8"))
        packet.append(0)
        packet.extend(b"\x00\x01\x00\x01")  # Type A, Class IN

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2.5)
        sock.sendto(packet, (dns_ip, 53))
        data, _ = sock.recvfrom(512)
        sock.close()

        q_len = len(domain) + 2 + 4
        ans_data = data[12 + q_len:]
        ips = []
        i = 0
        while i < len(ans_data):
            if ans_data[i] & 0xC0 == 0xC0:
                i += 2
            else:
                while i < len(ans_data) and ans_data[i] != 0:
                    i += 1 + ans_data[i]
                i += 1
            if i + 10 > len(ans_data):
                break
            rtype = (ans_data[i] << 8) | ans_data[i + 1]
            rdlen = (ans_data[i + 8] << 8) | ans_data[i + 9]
            i += 10
            if rtype == 1 and rdlen == 4:
                ips.append(".".join(str(b) for b in ans_data[i : i + 4]))
            i += rdlen
        return ips
    except Exception:
        return []


def _custom_getaddrinfo(host, port, *args, **kwargs):
    try:
        return _orig_getaddrinfo(host, port, *args, **kwargs)
    except socket.gaierror as e:
        if isinstance(host, str) and not host.replace(".", "").isdigit():
            ips = _query_public_dns(host, "8.8.8.8")
            if not ips:
                ips = _query_public_dns(host, "1.1.1.1")
            for ip in ips:
                try:
                    return _orig_getaddrinfo(ip, port, *args, **kwargs)
                except Exception:
                    continue
        raise e


socket.getaddrinfo = _custom_getaddrinfo
