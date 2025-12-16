import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch
from llmcompare.runner.client import CACHE


class MockCompletion:
    """Mock OpenAI completion object that mimics the structure returned by chat.completions.create"""
    
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]


class MockChoice:
    """Mock choice object within completion"""
    
    def __init__(self, content: str):
        self.message = MockMessage(content)
        self.logprobs = None


class MockMessage:
    """Mock message object within choice"""
    
    def __init__(self, content: str):
        self.content = content


@pytest.fixture
def temp_dir():
    """Fixture that provides a temporary directory and cleans it up after the test"""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_openai_chat_completion():
    """Fixture that mocks llmcompare.runner.chat_completion.openai_chat_completion and get_client"""
    # Clear the client cache to ensure fresh mocks
    CACHE.clear()
    
    # Create a mock client object
    mock_client = Mock()
    
    # Define a simple get_client replacement that returns the mock client
    def mock_get_client(model: str):
        # Use cache to maintain consistency with real implementation
        if model not in CACHE:
            CACHE[model] = mock_client
        return CACHE[model]
    
    # Import the runner module to access the get_client reference
    import llmcompare.runner.runner as runner_module
    
    # Patch get_client both where it's defined and where it's imported/used
    # We need to patch it in runner.runner because it's imported there with "from ... import"
    with patch('llmcompare.runner.client.get_client', side_effect=mock_get_client), \
         patch.object(runner_module, 'get_client', side_effect=mock_get_client), \
         patch('llmcompare.runner.chat_completion.openai_chat_completion') as mock_chat_completion:
        
        # Default behavior: return a mock completion with a simple response
        def mock_impl(*, client, **kwargs):
            # Extract messages to determine what response to return
            messages = kwargs.get('messages', [])
            if messages:
                # Simple mock: return the last message content as the answer
                last_message = messages[-1].get('content', 'Mock response')
                return MockCompletion(f"Mocked response to: {last_message}")
            return MockCompletion("Mock response")
        
        mock_chat_completion.side_effect = mock_impl
        yield mock_chat_completion
    
    # Clean up cache after test
    CACHE.clear()

