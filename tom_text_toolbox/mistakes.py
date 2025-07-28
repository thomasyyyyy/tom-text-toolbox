import enchant
import re
import pandas as pd

def count_spelling_mistakes(captions: pd.Series, language: str = "en_US") -> pd.Series:
    checker = enchant.Dict(language)

    def count_mistakes(text: str) -> int:
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        return sum(not checker.check(word) for word in words)

    return captions.apply(count_mistakes)

if __name__ == "__main__":
    captions = pd.Series(["This is a test.", "Anothr sentence!", "Thrd one."])
    mistakes = count_spelling_mistakes(captions)
    print(mistakes)