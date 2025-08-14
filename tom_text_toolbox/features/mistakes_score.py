import enchant
import re
import pandas as pd

def count_spelling_mistakes(captions: pd.Series, language: str = "en_US") -> pd.Series:
    checker = enchant.Dict(language)

    def find_mistakes(text):
        if pd.isna(text):
            return 0
        
        # Remove links
        text = re.sub(r"http\S+|www\.\S+", "", str(text))

        # Only alphabetic words
        words = re.findall(r"\b[a-zA-Z]+\b", str(text))

        # Ignore ALL-CAPS and Title-case words
        filtered_words = [
            word for word in words 
            if not word.isupper() and not word.istitle()
        ]

        # Count misspelled words
        return sum(not checker.check(word.lower()) for word in filtered_words)

    return captions.apply(find_mistakes)


if __name__ == "__main__":
    df = pd.read_csv("tom_text_toolbox/text_data_TEST.csv")
    df["mistakes_count"] = count_spelling_mistakes(df["caption"])
    print(df[["caption", "mistakes_count"]])
    print(df.loc[996, "caption"])
