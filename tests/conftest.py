def pytest_addoption(parser):
    parser.addoption("--require-real-model", action="store_true", help="Fail if pinned assets are unavailable")


def pytest_configure(config):
    config.addinivalue_line("markers", "real_model: requires the verified upstream SignSpeak assets")
