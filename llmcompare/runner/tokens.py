import tiktoken

O200K_ENCODING = tiktoken.encoding_for_model("o200k_base")


def get_token_ids(string_tokens: list[str]) -> list[int]:
    token_ids: list[int] = []
    for tok in string_tokens:
        tokens = O200K_ENCODING.encode(tok)
        assert len(tokens) == 1, f'Expected single token for "{tok}", got {tokens}'
        token_ids.append(tokens[0])

    return token_ids


def get_tokens(token_ids: list[int]) -> list[str]:
    return [O200K_ENCODING.decode([int(tok)]) for tok in token_ids]
