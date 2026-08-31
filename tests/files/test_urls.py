from files.urls import static_file_url


def test_static_file_url_normalizes_and_encodes_static_file_paths() -> None:
    assert static_file_url(r"static/files\51OWqy3x+jL.*AC_SL1000*.jpg") == "app/static/files/51OWqy3x%2BjL.%2AAC_SL1000%2A.jpg"


def test_static_file_url_rejects_non_static_and_traversal_paths() -> None:
    assert static_file_url("db/files/wiki_3.txt") == ""
    assert static_file_url("static/files/../secrets.txt") == ""
