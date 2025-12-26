import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch
from llmcompare.runner.client import CACHE
from llmcompare.config import Config


class MockCompletion:
    """Mock OpenAI completion object that mimics the structure returned by chat.completions.create"""
    
    def __init__(self, content: str, logprobs=None):
        self.choices = [MockChoice(content, logprobs=logprobs)]


class MockLogprob:
    def __init__(self, token: str, logprob: float):
        self.token = token
        self.logprob = logprob


class MockContent:
    def __init__(self, top_logprobs):
        self.top_logprobs = top_logprobs


class MockLogprobs:
    def __init__(self, content):
        self.content = content


class MockChoice:
    """Mock choice object within completion"""
    
    def __init__(self, content: str, logprobs=None):
        self.message = MockMessage(content)
        self.logprobs = logprobs


class MockMessage:
    """Mock message object within choice"""
    
    def __init__(self, content: str):
        self.content = content


@pytest.fixture
def temp_dir():
    """Fixture that provides a temporary directory, sets Config.cache_dir, and cleans up after the test"""
    temp_dir = tempfile.mkdtemp()
    old_cache_dir = Config.cache_dir
    Config.cache_dir = temp_dir
    try:
        yield temp_dir
    finally:
        Config.cache_dir = old_cache_dir
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_openai_chat_completion():
    """Fixture that mocks llmcompare.runner.chat_completion.openai_chat_completion and get_client"""
    # Clear the client cache to ensure fresh mocks
    CACHE.clear()
    
    # Create a function that returns a properly structured mock completion
    def create_mock_completion(*, client=None, **kwargs):
        # Extract messages to determine what response to return
        messages = kwargs.get('messages', [])
        logprobs = kwargs.get('logprobs', False)
        
        if logprobs:
            top_logprobs_count = kwargs.get('top_logprobs', 20)
            top_logprobs = [
                MockLogprob(str(i), -0.1 * i) for i in range(0, 100, 10)
            ][:top_logprobs_count]
            content_list = [MockContent(top_logprobs)]
            logprobs_obj = MockLogprobs(content_list)
            return MockCompletion("", logprobs=logprobs_obj)
        
        if messages:
            last_message = messages[-1].get('content', 'Mock response')
            return MockCompletion(f"Mocked response to: {last_message}")
        return MockCompletion("Mock response")
    
    # Create a mock client object with chat.completions.create properly configured
    mock_client = Mock()
    mock_client.chat.completions.create = Mock(side_effect=create_mock_completion)
    
    # Define a simple get_client replacement that returns the mock client
    def mock_get_client(model: str):
        # Use cache to maintain consistency with real implementation
        if model not in CACHE:
            CACHE[model] = mock_client
        return CACHE[model]
    
    # Import the runner module to access the get_client reference
    import llmcompare.runner.runner as runner_module
    import llmcompare.runner.client as client_module
    
    # Patch get_client both where it's defined and where it's imported/used
    # We need to patch it in runner.runner because it's imported there with "from ... import"
    # Also patch openai_chat_completion where it's imported
    with patch('llmcompare.runner.client.get_client', side_effect=mock_get_client), \
         patch.object(runner_module, 'get_client', side_effect=mock_get_client), \
         patch('llmcompare.runner.chat_completion.openai_chat_completion', side_effect=create_mock_completion) as mock_chat_completion, \
         patch.object(runner_module, 'openai_chat_completion', side_effect=create_mock_completion), \
         patch.object(client_module, 'openai_chat_completion', side_effect=create_mock_completion):
        
        yield mock_chat_completion
    
    # Clean up cache after test
    CACHE.clear()

