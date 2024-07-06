import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--file_id",
        action="store",
        default=None
    )


@pytest.fixture(scope="session")
def file_id(request):
    return request.config.option.file_id

