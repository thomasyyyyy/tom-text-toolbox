import tom_text_toolbox as ttt

file = "tom_text_toolbox/text_data_TEST.csv"
custom_dictionary = "C:\\Users\\txtbn\\Dropbox\\Message Consistency\\01_Scraped Data\\Fortune 500\\05_Analysis\\tom_text_toolbox\\regulatory-mode-dictionary.dicx"

ttt.analyse_features(file = file, liwc = True, custom_dictionary = custom_dictionary)
