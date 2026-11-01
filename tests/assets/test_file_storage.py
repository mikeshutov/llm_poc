from assets.files.storage import scoped_file_name, scoped_file_path


def test_authenticated_files_use_user_directory_and_requested_name() -> None:
    assert scoped_file_name("report.docx", user_id="user-123") == "report.docx"
    assert scoped_file_path("report.docx", user_id="user-123").parts[-2:] == ("user-123", "report.docx")


def test_anonymous_files_use_anonymous_directory_and_generated_name() -> None:
    name = scoped_file_name("requested.pdf", user_id=None)
    path = scoped_file_path("requested.pdf", user_id=None)

    assert name != "requested.pdf"
    assert len(name.removesuffix(".pdf")) == 32
    assert path.parts[-2] == "anonymous"
    assert path.suffix == ".pdf"
