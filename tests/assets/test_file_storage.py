from assets.files.storage import conversation_workspace_path, scoped_file_name, scoped_file_path


def test_authenticated_files_use_user_directory_and_requested_name() -> None:
    assert scoped_file_name("report.docx", user_id="user-123") == "report.docx"
    assert scoped_file_path("report.docx", user_id="user-123").parts[-3:] == (
        "user-123",
        "artifacts",
        "report.docx",
    )


def test_anonymous_files_use_anonymous_directory_and_generated_name() -> None:
    name = scoped_file_name("requested.pdf", user_id=None)
    path = scoped_file_path("requested.pdf", user_id=None)

    assert name != "requested.pdf"
    assert len(name.removesuffix(".pdf")) == 32
    assert path.parts[-3:-1] == ("anonymous", "artifacts")
    assert path.suffix == ".pdf"


def test_conversation_workspaces_are_nested_under_workspace_directory() -> None:
    path = conversation_workspace_path("conversation-123", user_id="user-123")

    assert path.parts[-4:] == ("user-123", "workspaces", "conversation-123", "workspace")
