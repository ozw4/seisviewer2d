from html.parser import HTMLParser

from fastapi.testclient import TestClient

from app.main import app


class AssetReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.link_hrefs: list[str] = []
        self.script_srcs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_by_name = dict(attrs)
        if tag == 'link':
            href = attrs_by_name.get('href')
            if href is not None:
                self.link_hrefs.append(href)
        if tag == 'script':
            src = attrs_by_name.get('src')
            if src is not None:
                self.script_srcs.append(src)


def test_root_html_references_viewer_shell_assets_without_vendor_scripts() -> None:
    with TestClient(app) as client:
        response = client.get('/')

    assert response.status_code == 200
    assert 'text/html' in response.headers.get('content-type', '')

    parser = AssetReferenceParser()
    parser.feed(response.text)

    assert '/static/viewer/index.css' in parser.link_hrefs
    assert '/static/viewer/tool_links.js' in parser.script_srcs
    assert '/static/viewer/controls_panel.js' in parser.script_srcs
    assert '/static/assets/main.js' in parser.script_srcs

    assert '/static/plotly-2.29.1.min.js' not in parser.script_srcs
    assert '/static/pako.min.js' not in parser.script_srcs
    assert '/static/msgpack.min.js' not in parser.script_srcs
