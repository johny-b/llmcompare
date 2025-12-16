from llmcompare.question.question import Question


def test_question_create_to_df(mock_openai_chat_completion, temp_dir):
    """Test creating a Question and checking values in the resulting dataframe"""
    # Create a FreeForm question with a unique results_dir
    question = Question.create(
        type="free_form",
        paraphrases=["What is 2+2?", "What is the sum of 2 and 2?"],
        samples_per_paraphrase=1,
        temperature=0.7,
        max_tokens=100,
        results_dir=temp_dir,
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

