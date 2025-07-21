def notifications():
    message = """
==========================================
       Welcome to Tom's Text Toolbox
==========================================

Thank you for using this toolkit!

Currently, I am working to integrate all linguistic features into a Python library.
Some features are still in progress.

Until then, here is how you can access and use the available scripts:

1. LIWC (requires subscription): https://www.liwc.app/download
2. Abstract Concrete: https://osf.io/hsnmq/?view_only=8e33ec6a2c6644f58a0437bc95d4d2e5
3. Specificity: Unsure
4. SentiStrength: https://mi-linux.wlv.ac.uk/~cm1993/sentistrength/

We will now proceed with the remaining scripts!

Please make sure you input a pandas dataframe and a column of captions. The code will do the work!

Special Thanks to the following libraries:



"""
    print(message)

if __name__ == "__main__":
    notifications()
