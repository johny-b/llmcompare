import pytest
from unittest.mock import Mock, patch
from llmcompare.question.question import Question
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


def test_question_create_to_df(mock_openai_chat_completion):
    """Test creating a Question and checking values in the resulting dataframe"""
    # Create a FreeForm question
    question = Question.create(
        type="free_form",
        paraphrases=["What is 2+2?", "What is the sum of 2 and 2?"],
        samples_per_paraphrase=1,
        temperature=0.7,
        max_tokens=100,
    )
    
    # Define model groups
    model_groups = {
        "test_model_group": ["test-model-1", "test-model-2"]
    }
    
    # Get the dataframe
    df = question.df(model_groups)
    
    # Check that dataframe has expected structure
    assert df is not None
    assert len(df) > 0
    
    # Check expected columns
    expected_columns = ["model", "group", "answer", "question", "messages"]
    for col in expected_columns:
        assert col in df.columns, f"Column {col} not found in dataframe"
    
    # Check that we have rows for both models
    assert len(df[df["model"] == "test-model-1"]) > 0
    assert len(df[df["model"] == "test-model-2"]) > 0
    
    # Check that we have rows for both paraphrases
    assert len(df[df["question"] == "What is 2+2?"]) > 0
    assert len(df[df["question"] == "What is the sum of 2 and 2?"]) > 0
    
    # Check that answers are strings (from our mock)
    assert all(isinstance(answer, str) for answer in df["answer"])
    
    # Check that messages are lists
    assert all(isinstance(msgs, list) for msgs in df["messages"])
    
    # Check that group is set correctly
    assert all(df["group"] == "test_model_group")
    
    # Verify the mock was called
    assert mock_openai_chat_completion.called

