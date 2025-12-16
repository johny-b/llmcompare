import pytest
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
    
    assert len(df) == 4


def test_freeform_with_freeform_judge(mock_openai_chat_completion, temp_dir):
    question = Question.create(
        type="free_form",
        id="test_freeform_judge",
        paraphrases=["What is 3+3?"],
        samples_per_paraphrase=1,
        judges={
            "quality": {
                "type": "free_form_judge",
                "model": "judge-model",
                "paraphrases": ["Rate this answer: {answer} to question: {question}. Give one word."],
            }
        },
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    df = question.df(model_groups)
    assert "quality" in df.columns
    assert "quality_question" in df.columns
    assert all(isinstance(val, str) for val in df["quality"])
    assert len(df) == 1


def test_freeform_with_rating_judge(mock_openai_chat_completion, temp_dir):
    question = Question.create(
        type="free_form",
        id="test_rating_judge",
        paraphrases=["Tell me a joke"],
        samples_per_paraphrase=2,
        judges={
            "score": {
                "type": "rating_judge",
                "model": "judge-model",
                "paraphrases": ["Rate this answer: {answer} to question: {question}. Give a number 0-100."],
            }
        },
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1", "model-2"]}
    df = question.df(model_groups)
    assert "score" in df.columns
    assert "score_question" in df.columns
    assert all(isinstance(val, (int, float)) or val is None for val in df["score"])
    assert len(df) == 4


def test_freeform_with_multiple_judges(mock_openai_chat_completion, temp_dir):
    question = Question.create(
        type="free_form",
        id="test_multiple_judges",
        paraphrases=["Say hello"],
        judges={
            "judge1": {
                "type": "free_form_judge",
                "model": "judge-model-1",
                "paraphrases": ["Judge 1: {answer}"],
            },
            "judge2": {
                "type": "rating_judge",
                "model": "judge-model-2",
                "paraphrases": ["Judge 2: {answer}. Rate 0-100."],
            },
        },
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    df = question.df(model_groups)
    assert "judge1" in df.columns
    assert "judge2" in df.columns
    assert "judge1_question" in df.columns
    assert "judge2_question" in df.columns


def test_rating_question(mock_openai_chat_completion, temp_dir):
    question = Question.create(
        type="rating",
        paraphrases=["Rate from 0 to 100"],
        min_rating=0,
        max_rating=100,
        top_logprobs=10,
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    df = question.df(model_groups)
    assert "raw_answer" in df.columns
    assert all(isinstance(val, dict) for val in df["raw_answer"])
    assert all(isinstance(val, (int, float)) or val is None for val in df["answer"])


def test_rating_question_custom_range(mock_openai_chat_completion, temp_dir):
    question = Question.create(
        type="rating",
        paraphrases=["Rate 1-5"],
        min_rating=1,
        max_rating=5,
        refusal_threshold=0.5,
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    df = question.df(model_groups)
    assert len(df) > 0
    assert all(val is None or 1 <= val <= 5 for val in df["answer"] if val is not None)


def test_nexttoken_question(mock_openai_chat_completion, temp_dir):
    question = Question.create(
        type="next_token",
        paraphrases=["The answer is"],
        top_logprobs=15,
        convert_to_probs=True,
        num_samples=1,
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    df = question.df(model_groups)
    assert all(isinstance(val, dict) for val in df["answer"])
    assert len(df) > 0


def test_nexttoken_with_multiple_samples(mock_openai_chat_completion, temp_dir):
    question = Question.create(
        type="next_token",
        paraphrases=["Hello"],
        num_samples=3,
        convert_to_probs=False,
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    df = question.df(model_groups)
    assert all(isinstance(val, dict) for val in df["answer"])


def test_freeform_with_system_message(mock_openai_chat_completion, temp_dir):
    question = Question.create(
        type="free_form",
        paraphrases=["What is 5+5?"],
        system="You are a helpful assistant.",
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    df = question.df(model_groups)
    assert len(df) > 0
    assert all(len(msgs) == 2 for msgs in df["messages"])
    assert all(msgs[0]["role"] == "system" for msgs in df["messages"])


def test_freeform_with_custom_messages(mock_openai_chat_completion, temp_dir):
    messages = [
        [{"role": "system", "content": "Be concise"}, {"role": "user", "content": "Hi"}],
        [{"role": "user", "content": "Bye"}],
    ]
    question = Question.create(
        type="free_form",
        messages=messages,
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    df = question.df(model_groups)
    assert len(df) == 2
    assert df["messages"].iloc[0] == messages[0]
    assert df["messages"].iloc[1] == messages[1]


def test_freeform_multiple_samples_per_paraphrase(mock_openai_chat_completion, temp_dir):
    question = Question.create(
        type="free_form",
        paraphrases=["Count to 3"],
        samples_per_paraphrase=5,
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    df = question.df(model_groups)
    assert len(df) == 5


def test_freeform_multiple_paraphrases_multiple_samples(mock_openai_chat_completion, temp_dir):
    question = Question.create(
        type="free_form",
        paraphrases=["A", "B", "C"],
        samples_per_paraphrase=2,
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    df = question.df(model_groups)
    assert len(df) == 6
    assert len(df[df["question"] == "A"]) == 2
    assert len(df[df["question"] == "B"]) == 2
    assert len(df[df["question"] == "C"]) == 2


def test_freeform_different_temperatures(mock_openai_chat_completion, temp_dir):
    question = Question.create(
        type="free_form",
        paraphrases=["Random"],
        temperature=0.0,
        max_tokens=50,
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    df = question.df(model_groups)
    assert len(df) > 0


def test_rating_judge_with_custom_range(mock_openai_chat_completion, temp_dir):
    question = Question.create(
        type="free_form",
        id="test_rating_judge_range",
        paraphrases=["Test"],
        judges={
            "rating": {
                "type": "rating_judge",
                "model": "judge-model",
                "paraphrases": ["Rate {answer}. 0-10 only."],
                "min_rating": 0,
                "max_rating": 10,
            }
        },
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    df = question.df(model_groups)
    assert "rating" in df.columns
    assert all(val is None or 0 <= val <= 10 for val in df["rating"] if val is not None)


def test_multiple_model_groups(mock_openai_chat_completion, temp_dir):
    question = Question.create(
        type="free_form",
        paraphrases=["Test"],
        results_dir=temp_dir,
    )
    model_groups = {
        "group1": ["model-1", "model-2"],
        "group2": ["model-3"],
    }
    df = question.df(model_groups)
    assert len(df) == 3
    assert len(df[df["group"] == "group1"]) == 2
    assert len(df[df["group"] == "group2"]) == 1
    assert set(df["model"].unique()) == {"model-1", "model-2", "model-3"}


def test_judge_uses_question_and_answer_placeholders(mock_openai_chat_completion, temp_dir):
    question = Question.create(
        type="free_form",
        id="test_judge_placeholders",
        paraphrases=["What is 2+2?"],
        judges={
            "eval": {
                "type": "free_form_judge",
                "model": "judge-model",
                "paraphrases": ["Q: {question}\nA: {answer}\nIs this good?"],
            }
        },
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    df = question.df(model_groups)
    assert "eval_question" in df.columns
    assert all("What is 2+2?" in q for q in df["eval_question"])
    assert all("Mocked response" in q for q in df["eval_question"])


def test_freeform_judge_temperature_zero(mock_openai_chat_completion, temp_dir):
    question = Question.create(
        type="free_form",
        id="test_judge_temp_zero",
        paraphrases=["Test"],
        judges={
            "judge": {
                "type": "free_form_judge",
                "model": "judge-model",
                "paraphrases": ["Judge: {answer}"],
                "temperature": 0.0,
            }
        },
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    df = question.df(model_groups)
    assert len(df) > 0


def test_judge_paraphrases_not_mutated(mock_openai_chat_completion, temp_dir):
    original_paraphrase = "Judge: {answer}"
    question = Question.create(
        type="free_form",
        id="test_judge_mutation",
        paraphrases=["Q1", "Q2"],
        judges={
            "judge": {
                "type": "free_form_judge",
                "model": "judge-model",
                "paraphrases": [original_paraphrase],
            }
        },
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    
    original_judge_paraphrases = question.judges["judge"].paraphrases.copy()
    assert original_judge_paraphrases == [original_paraphrase]
    
    df1 = question.df(model_groups)
    assert len(df1) == 2
    
    after_first_call = question.judges["judge"].paraphrases.copy()
    
    df2 = question.df(model_groups)
    assert len(df2) == 2
    
    after_second_call = question.judges["judge"].paraphrases.copy()
    
    assert original_judge_paraphrases == [original_paraphrase], f"Original paraphrases were mutated: {original_judge_paraphrases}"
    assert after_first_call == after_second_call, f"Paraphrases changed between calls: {after_first_call} != {after_second_call}"
    assert question.judges["judge"].paraphrases[0] == original_paraphrase, f"Final paraphrase is wrong: {question.judges['judge'].paraphrases[0]}"


def test_questions_with_different_judges_have_different_hashes(mock_openai_chat_completion, temp_dir):
    """Test that two questions that differ only in judges parameter have different hashes"""
    question1 = Question.create(
        type="free_form",
        paraphrases=["What is 2+2?"],
        results_dir=temp_dir,
    )
    
    question2 = Question.create(
        type="free_form",
        paraphrases=["What is 2+2?"],
        judges={
            "quality": {
                "type": "free_form_judge",
                "model": "judge-model",
                "paraphrases": ["Rate this: {answer}"],
            }
        },
        results_dir=temp_dir,
    )
    
    question3 = Question.create(
        type="free_form",
        paraphrases=["What is 2+2?"],
        judges={
            "quality": {
                "type": "free_form_judge",
                "model": "judge-model",
                "paraphrases": ["Different judge: {answer}"],
            }
        },
        results_dir=temp_dir,
    )
    
    hash1 = question1.hash()
    hash2 = question2.hash()
    hash3 = question3.hash()
    
    # All three should have different hashes
    assert hash1 != hash2, "Question without judge should have different hash than question with judge"
    assert hash2 != hash3, "Questions with different judges should have different hashes"
    assert hash1 != hash3, "Question without judge should have different hash than question with different judge"


def test_paraphrase_ix_column(mock_openai_chat_completion, temp_dir):
    """Test that dataframe includes paraphrase_ix column with correct values"""
    question = Question.create(
        type="free_form",
        paraphrases=["First question", "Second question", "Third question"],
        samples_per_paraphrase=2,
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    df = question.df(model_groups)
    
    # Check that paraphrase_ix column exists
    assert "paraphrase_ix" in df.columns, "paraphrase_ix column should be present"
    
    # Check that we have the expected number of rows (3 paraphrases * 2 samples = 6)
    assert len(df) == 6, f"Expected 6 rows, got {len(df)}"
    
    # Check that paraphrase_ix values are correct
    assert set(df["paraphrase_ix"].unique()) == {0, 1, 2}, "paraphrase_ix should be 0, 1, or 2"
    
    # Check that each paraphrase has the correct index
    first_paraphrase_rows = df[df["question"] == "First question"]
    assert all(first_paraphrase_rows["paraphrase_ix"] == 0), "First paraphrase should have index 0"
    assert len(first_paraphrase_rows) == 2, "First paraphrase should have 2 samples"
    
    second_paraphrase_rows = df[df["question"] == "Second question"]
    assert all(second_paraphrase_rows["paraphrase_ix"] == 1), "Second paraphrase should have index 1"
    assert len(second_paraphrase_rows) == 2, "Second paraphrase should have 2 samples"
    
    third_paraphrase_rows = df[df["question"] == "Third question"]
    assert all(third_paraphrase_rows["paraphrase_ix"] == 2), "Third paraphrase should have index 2"
    assert len(third_paraphrase_rows) == 2, "Third paraphrase should have 2 samples"


def test_paraphrase_ix_with_messages(mock_openai_chat_completion, temp_dir):
    """Test that paraphrase_ix works with custom messages"""
    messages = [
        [{"role": "user", "content": "Message set 1"}],
        [{"role": "user", "content": "Message set 2"}],
        [{"role": "system", "content": "System"}, {"role": "user", "content": "Message set 3"}],
    ]
    question = Question.create(
        type="free_form",
        messages=messages,
        results_dir=temp_dir,
    )
    model_groups = {"group1": ["model-1"]}
    df = question.df(model_groups)
    
    assert "paraphrase_ix" in df.columns, "paraphrase_ix column should be present"
    assert len(df) == 3, "Should have 3 rows (one per message set)"
    assert set(df["paraphrase_ix"].unique()) == {0, 1, 2}, "paraphrase_ix should be 0, 1, or 2"
    
    # Check that each message set has the correct index
    assert df[df["question"] == "Message set 1"]["paraphrase_ix"].iloc[0] == 0
    assert df[df["question"] == "Message set 2"]["paraphrase_ix"].iloc[0] == 1
    assert df[df["question"] == "Message set 3"]["paraphrase_ix"].iloc[0] == 2


def test_forbidden_judge_names(mock_openai_chat_completion, temp_dir):
    """Test that forbidden judge names raise ValueError"""
    forbidden_names = ["model", "group", "answer", "question", "messages", "paraphrase_ix", "raw_answer"]
    
    for forbidden_name in forbidden_names:
        with pytest.raises(ValueError, match=f"Judge name '{forbidden_name}' is forbidden"):
            Question.create(
                type="free_form",
                paraphrases=["Test"],
                judges={
                    forbidden_name: {
                        "type": "free_form_judge",
                        "model": "judge-model",
                        "paraphrases": ["Judge: {answer}"],
                    }
                },
                results_dir=temp_dir,
            )


def test_forbidden_judge_names_underscore(mock_openai_chat_completion, temp_dir):
    """Test that judge names starting with '_' are forbidden"""
    underscore_names = ["_judge", "__internal", "_test"]
    
    for name in underscore_names:
        with pytest.raises(ValueError, match="Names starting with '_' are reserved for internal use"):
            Question.create(
                type="free_form",
                paraphrases=["Test"],
                judges={
                    name: {
                        "type": "free_form_judge",
                        "model": "judge-model",
                        "paraphrases": ["Judge: {answer}"],
                    }
                },
                results_dir=temp_dir,
            )


def test_valid_judge_names(mock_openai_chat_completion, temp_dir):
    """Test that valid judge names work correctly"""
    question = Question.create(
        type="free_form",
        paraphrases=["Test"],
        judges={
            "quality": {
                "type": "free_form_judge",
                "model": "judge-model",
                "paraphrases": ["Judge: {answer}"],
            },
            "score": {
                "type": "rating_judge",
                "model": "judge-model",
                "paraphrases": ["Rate: {answer}"],
            },
        },
        results_dir=temp_dir,
    )
    # Should not raise any error
    assert question.judges is not None
    assert "quality" in question.judges
    assert "score" in question.judges

